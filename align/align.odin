// Forced alignment: where each word of a known transcript is spoken in a
// recording, from a CTC speech recognizer's per-frame letter
// probabilities (wav2vec2-base-960h: English letters and the apostrophe,
// 20 ms frames at 16 kHz) and a Viterbi path through the transcript's
// letters (as torchaudio's forced_align).
//
// A word the recognizer cannot spell ("3", "50", "€") is a wildcard: it
// matches whatever the frames hold, so "$3.50" said as "three fifty" or
// "three dollars and fifty cents" lands between its neighbours. Runs of
// such words share one wildcard and split its span by length.
package align

import "core:fmt"
import "core:math"
import "core:strings"
import "core:unicode/utf8"
import "../ort"

SAMPLE_RATE :: 16000
FRAME_SAMPLES :: 320 // the recognizer's stride: 20 ms

// wav2vec2-base-960h's vocabulary (vocab.json): <pad> is the CTC blank.
BLANK :: 0
WORD_GAP :: 4 // "|"
WILDCARD :: -1

@(private)
letter_id :: proc(c: u8) -> int {
	// E T A O N I H S R D L U M W C F G Y P B V K ' X J Q Z
	@(static, rodata) ids := [26]int{7, 24, 19, 14, 5, 20, 21, 11, 10, 29, 26, 15, 17, 9, 8, 23, 30, 13, 12, 6, 16, 25, 18, 28, 22, 31}
	switch c {
	case 'A' ..= 'Z': return ids[c - 'A']
	case 'a' ..= 'z': return ids[c - 'a']
	case '\'': return 27
	}
	return -1
}

// The recognizer's letters for a word: ASCII letters and apostrophes,
// Latin accents folded ("café" -> CAFE), anything else dropped.
@(private)
spell :: proc(word: string, out: ^[dynamic]int) {
	for r in word {
		folded: string
		switch r {
		case 'à', 'á', 'â', 'ã', 'ä', 'å', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å': folded = "A"
		case 'ç', 'Ç': folded = "C"
		case 'è', 'é', 'ê', 'ë', 'È', 'É', 'Ê', 'Ë': folded = "E"
		case 'ì', 'í', 'î', 'ï', 'Ì', 'Í', 'Î', 'Ï': folded = "I"
		case 'ñ', 'Ñ': folded = "N"
		case 'ò', 'ó', 'ô', 'õ', 'ö', 'ø', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'Ø': folded = "O"
		case 'ù', 'ú', 'û', 'ü', 'Ù', 'Ú', 'Û', 'Ü': folded = "U"
		case 'ý', 'ÿ', 'Ý': folded = "Y"
		case 'æ', 'Æ': folded = "AE"
		case 'œ', 'Œ': folded = "OE"
		case 'ß': folded = "SS"
		case '’', '‘', '`': folded = "'"
		case:
			if r < 0x80 {
				buf, n := utf8.encode_rune(r)
				folded = string(buf[:n])
				folded = strings.clone(folded, context.temp_allocator)
			}
		}
		for i in 0 ..< len(folded) {
			if id := letter_id(folded[i]); id >= 0 do append(out, id)
		}
	}
}

Aligner :: struct {
	model:  ort.Model,
	logits: int, // output index
}

init :: proc(a: ^Aligner, rt: ^ort.Runtime, path: string) -> ort.Error {
	a.model = ort.load_model(rt, path) or_return
	if len(a.model.inputs) != 1 || string(a.model.inputs[0]) != "input_values" do return "aligner: expected one input, input_values"
	a.logits = ort.output_index(&a.model, "logits")
	if a.logits < 0 do return "aligner: no logits output"
	return nil
}

destroy :: proc(a: ^Aligner) {
	ort.unload_model(&a.model)
}

Span :: struct {
	start, end: f64, // seconds from the start of the audio
	aligned:    bool, // false: the word was a wildcard, its span shared by length
}

// Where each of `words` is spoken in `audio` (mono, `rate` Hz), in order.
// Fails when the audio is too short for the transcript's letters.
align :: proc(a: ^Aligner, audio: []f32, rate: int, words: []string, spans: []Span) -> ort.Error {
	assert(len(spans) == len(words))
	if len(words) == 0 do return nil
	rt := a.model.rt

	// Recognizer input: 16 kHz, zero mean, unit variance.
	x := resample(audio, rate, SAMPLE_RATE, context.temp_allocator)
	if len(x) < 400 do return "aligner: audio too short"
	mean, sq: f64
	for v in x do mean += f64(v)
	mean /= f64(len(x))
	for v in x do sq += (f64(v) - mean) * (f64(v) - mean)
	scale := 1 / math.sqrt(sq / f64(len(x)) + 1e-7)
	for &v in x do v = f32((f64(v) - mean) * scale)

	input := ort.tensor_f32(rt, x, {1, i64(len(x))}) or_return
	defer ort.release(rt, input)
	outputs := make([]^ort.Value, len(a.model.outputs), context.temp_allocator)
	ort.run(&a.model, {input}, outputs) or_return
	defer for v in outputs do ort.release(rt, v)
	shape: [4]i64
	view := ort.view(rt, outputs[a.logits], shape[:]) or_return
	if len(view.shape) != 3 || view.shape[2] != 32 do return "aligner: logits are not [1, frames, 32]"
	frames := int(view.shape[1])
	logp := log_softmax(ort.view_f32(view), frames, 32)

	// Labels: each word's letters (or one wildcard for a run of
	// unspellable words), "|" between words.
	labels := make([dynamic]int, context.temp_allocator)
	label_word := make([dynamic]int, context.temp_allocator) // first word of each label's group
	group_first := make([dynamic]int, context.temp_allocator) // per group: first word
	group_count := make([dynamic]int, context.temp_allocator)
	letters := make([dynamic]int, context.temp_allocator)
	for w := 0; w < len(words); {
		clear(&letters)
		spell(words[w], &letters)
		first, count := w, 1
		if len(letters) == 0 {
			for w + count < len(words) {
				clear(&letters)
				spell(words[w + count], &letters)
				if len(letters) > 0 do break
				count += 1
			}
			clear(&letters)
			append(&letters, WILDCARD)
		}
		group := len(group_first)
		append(&group_first, first)
		append(&group_count, count)
		if len(labels) > 0 {
			append(&labels, WORD_GAP)
			append(&label_word, -1)
		}
		for l in letters {
			append(&labels, l)
			append(&label_word, group)
		}
		w += count
	}

	path, ok := viterbi(logp, frames, labels[:])
	if !ok do return fmt.aprintf("aligner: %d frames cannot hold %d letters", frames, len(labels))

	// Each group spans the frames its letters hold.
	seconds_per_frame := f64(len(audio)) / f64(rate) / f64(frames)
	first_frame := make([]int, len(group_first), context.temp_allocator)
	last_frame := make([]int, len(group_first), context.temp_allocator)
	for &f in first_frame do f = -1
	for state, t in path {
		if state % 2 == 0 do continue // blank
		g := label_word[state / 2]
		if g < 0 do continue
		if first_frame[g] < 0 do first_frame[g] = t
		last_frame[g] = t
	}
	for g in 0 ..< len(group_first) {
		start := f64(first_frame[g]) * seconds_per_frame
		end := f64(last_frame[g] + 1) * seconds_per_frame
		n := group_count[g]
		if n == 1 {
			spans[group_first[g]] = {start, end, labels_of_group_are_letters(labels[:], label_word[:], g)}
			continue
		}
		// A shared wildcard: split by the words' lengths.
		total := 0
		for k in 0 ..< n do total += max(utf8.rune_count_in_string(words[group_first[g] + k]), 1)
		at := start
		for k in 0 ..< n {
			share := f64(max(utf8.rune_count_in_string(words[group_first[g] + k]), 1)) / f64(total)
			spans[group_first[g] + k] = {at, at + (end - start) * share, false}
			at += (end - start) * share
		}
	}
	return nil
}

@(private)
labels_of_group_are_letters :: proc(labels, label_word: []int, g: int) -> bool {
	for l, i in labels do if label_word[i] == g && l == WILDCARD do return false
	return true
}

@(private)
log_softmax :: proc(logits: []f32, frames, classes: int) -> []f32 {
	out := make([]f32, frames * classes, context.temp_allocator)
	for t in 0 ..< frames {
		row := logits[t * classes:][:classes]
		m := row[0]
		for v in row do m = max(m, v)
		sum: f64
		for v in row do sum += math.exp(f64(v - m))
		lse := f32(math.ln(sum)) + m
		for v, k in row do out[t * classes + k] = v - lse
	}
	return out
}

// The most likely CTC path through `labels` (blanks between and around
// them, repeats needing a blank between): the state at each frame, where
// state 2k+1 is labels[k] and even states are blanks. A wildcard label
// scores each frame as its most likely non-blank class.
@(private)
viterbi :: proc(logp: []f32, frames: int, labels: []int) -> (path: []int, ok: bool) {
	states := 2 * len(labels) + 1
	NEG :: f32(-1e30)
	emit :: proc(logp: []f32, t: int, label: int) -> f32 {
		row := logp[t * 32:][:32]
		if label != WILDCARD do return row[label]
		best := NEG
		for v, k in row do if k != BLANK do best = max(best, v)
		return best
	}
	label_of :: proc(labels: []int, s: int) -> int {return BLANK if s % 2 == 0 else labels[s / 2]}

	prev := make([]f32, states, context.temp_allocator)
	cur := make([]f32, states, context.temp_allocator)
	back := make([]u8, frames * states, context.temp_allocator) // 0 stay, 1 from s-1, 2 from s-2
	for &v in prev do v = NEG
	prev[0] = emit(logp, 0, BLANK)
	if states > 1 do prev[1] = emit(logp, 0, labels[0])
	for t in 1 ..< frames {
		for s in 0 ..< states {
			best, from := prev[s], u8(0)
			if s >= 1 && prev[s - 1] > best do best, from = prev[s - 1], 1
			if s >= 2 && s % 2 == 1 {
				l := labels[s / 2]
				if (l == WILDCARD || l != labels[s / 2 - 1]) && prev[s - 2] > best do best, from = prev[s - 2], 2
			}
			cur[s] = best + emit(logp, t, label_of(labels, s)) if best > NEG else NEG
			back[t * states + s] = from
		}
		prev, cur = cur, prev
	}
	end := states - 1
	if states >= 2 && prev[states - 2] > prev[end] do end = states - 2
	if prev[end] <= NEG / 2 do return nil, false

	path = make([]int, frames, context.temp_allocator)
	s := end
	for t := frames - 1; t >= 0; t -= 1 {
		path[t] = s
		if t > 0 do s -= int(back[t * states + s])
	}
	return path, s <= 1
}
