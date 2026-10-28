package pocket

import "core:strings"
import "core:unicode"
import "core:unicode/utf8"
import "../spm"

// Text preparation and chunking as upstream Pocket TTS does it
// (pocket_tts/models/text_chunking.py @ b94db17): the model was trained on
// single sentences that start with a capital and end with . ! ? or …, so
// prompts are made to, and long text is split on sentence boundaries into
// chunks of at most `max_tokens` tokens, each generated on its own.

@(private) TERMINAL :: ".!?…"
@(private) WEAK :: ",;:-–—"
@(private) CLOSERS :: "\"'”’)]»"

// The prompt text for one chunk, and how many frames to keep after the
// model signals the end (more for very short prompts).
prepare_text :: proc(text: string, remove_semicolons: bool, allocator := context.allocator) -> (prepared: string, frames_after_eos: int, ok: bool) {
	context.allocator = allocator
	t := strings.trim_space(text)
	if t == "" do return "", 0, false
	t, _ = strings.replace_all(t, "\n", " ", context.temp_allocator)
	t, _ = strings.replace_all(t, "\r", " ", context.temp_allocator)
	t, _ = strings.replace_all(t, "  ", " ", context.temp_allocator)
	if remove_semicolons do t, _ = strings.replace_all(t, ";", ",", context.temp_allocator)

	words := 0
	rest := t
	for _ in strings.fields_iterator(&rest) do words += 1
	frames_after_eos = 3 if words <= 4 else 1

	first, n := utf8.decode_rune_in_string(t)
	if !unicode.is_upper(first) {
		t = strings.concatenate({utf8.runes_to_string({unicode.to_upper(first)}, context.temp_allocator), t[n:]}, context.temp_allocator)
	}
	return terminal_punctuation(t), frames_after_eos, true
}

// Ends the text with sentence-final punctuation, before any closing quotes
// or brackets; trailing weak punctuation becomes a period.
@(private)
terminal_punctuation :: proc(text: string) -> string {
	core := strings.trim_right(text, CLOSERS + " ")
	closers := strings.trim_space(text[len(core):])
	last, _ := utf8.decode_last_rune_in_string(core)
	if core == "" || strings.contains_rune(TERMINAL, last) do return strings.clone(text)
	if strings.contains_rune(WEAK, last) {
		return strings.concatenate({strings.trim_right(core, WEAK + " "), ".", closers})
	}
	return strings.concatenate({text, "."})
}

// Token positions where a new segment starts after a run of boundary
// tokens. With `skip_decimals`, a period between digits does not split.
@(private)
boundaries :: proc(tok: ^spm.Model, ids: []i32, boundary: []i32, skip_decimals: bool) -> [dynamic]int {
	out := make([dynamic]int, context.temp_allocator)
	append(&out, 0)
	// The decimal check reads the text on both sides of a position: decoded
	// once, with each token's offset in it.
	text: string
	starts: []int
	if skip_decimals {
		starts = make([]int, len(ids) + 1, context.temp_allocator)
		text = spm.decode(tok, ids, context.temp_allocator, starts)
	}
	previous := false
	for id, i in ids {
		is_boundary := false
		for b in boundary do is_boundary ||= id == b
		if is_boundary {
			previous = true
			continue
		}
		if previous {
			if skip_decimals && decimal_period(tok, ids, text, starts, i) {
				previous = false
				continue
			}
			append(&out, i)
		}
		previous = false
	}
	append(&out, len(ids))
	return out
}

// Whether a segment starting at token `at` (> 0) would start right after a
// decimal point: the text of the tokens before it ends in a digit and a
// period, and theirs from it on starts with a digit. Those are the
// decodes of ids[:at] and ids[at:] that upstream compares, read from
// `text`, all of `ids` decoded, at the token offsets `starts`; decoded on
// its own, ids[at:] loses its first piece's leading "▁" (a space here).
@(private)
decimal_period :: proc(tok: ^spm.Model, ids: []i32, text: string, starts: []int, at: int) -> bool {
	prefix, suffix := text[:starts[at]], text[starts[at]:]
	if id := ids[at]; tok.add_dummy_prefix && id >= 0 && int(id) < len(tok.pieces) {
		if (tok.types[id] == .Normal || tok.types[id] == .User_Defined) && strings.has_prefix(tok.pieces[id], spm.SPACE) do suffix = suffix[1:]
	}
	last, n := utf8.decode_last_rune_in_string(prefix)
	before, m := utf8.decode_last_rune_in_string(prefix[:len(prefix) - n])
	next, _ := utf8.decode_rune_in_string(suffix)
	return n > 0 && m > 0 && suffix != "" && last == '.' && unicode.is_digit(before) && unicode.is_digit(next)
}

@(private)
Segment :: struct {
	tokens: int,
	text:   string,
}

@(private)
segments :: proc(tok: ^spm.Model, ids: []i32, bounds: []int) -> [dynamic]Segment {
	out := make([dynamic]Segment, context.temp_allocator)
	for i in 0 ..< len(bounds) - 1 {
		a, b := bounds[i], bounds[i + 1]
		append(&out, Segment{b - a, spm.decode(tok, ids[a:b], context.temp_allocator)})
	}
	return out
}

@(private)
encode_ids :: proc(tok: ^spm.Model, text: string) -> []i32 {
	tokens := make([dynamic]spm.Token, context.temp_allocator)
	spm.encode(tok, text, &tokens)
	ids := make([]i32, len(tokens), context.temp_allocator)
	for t, i in tokens do ids[i] = t.id
	return ids
}

// Splits text into the chunks generated one at a time (in the temp allocator).
split_chunks :: proc(tok: ^spm.Model, text: string, max_tokens: int, remove_semicolons: bool) -> (chunks: [dynamic]string, ok: bool) {
	chunks = make([dynamic]string, context.temp_allocator)
	prepared, _ := prepare_text(text, remove_semicolons, context.temp_allocator) or_return
	prepared = strings.trim_space(prepared)
	ids := encode_ids(tok, prepared)

	sentence_ends := encode_ids(tok, ".!...?")[1:]
	sentences := segments(tok, ids, boundaries(tok, ids, sentence_ends, true)[:])

	// Oversized sentences split again on , ; :
	clause_ends := encode_ids(tok, ",;:")[1:]
	refined := make([dynamic]Segment, context.temp_allocator)
	for s in sentences {
		if s.tokens <= max_tokens {
			append(&refined, s)
			continue
		}
		sub := encode_ids(tok, strings.trim_space(s.text))
		parts := segments(tok, sub, boundaries(tok, sub, clause_ends, false)[:])
		if len(parts) > 1 {
			append(&refined, ..parts[:])
		} else {
			append(&refined, s)
		}
	}

	current := ""
	count := 0
	for s in refined {
		if current == "" {
			current, count = s.text, s.tokens
			continue
		}
		if count + s.tokens > max_tokens {
			append(&chunks, strings.trim_space(current))
			current, count = s.text, s.tokens
		} else {
			current = strings.concatenate({current, " ", s.text}, context.temp_allocator)
			count += s.tokens
		}
	}
	if current != "" do append(&chunks, strings.trim_space(current))
	return chunks, true
}
