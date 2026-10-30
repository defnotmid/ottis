// Pocket TTS (Kyutai, CALM) run through its ONNX export: text conditioner,
// the backbone (`flow_lm_main`), the one-step flow head (`flow_lm_flow`)
// and the Mimi decoder. The generation loop, sampling, end-of-speech rule
// and chunking follow upstream pocket_tts/models/tts_model.py @ b94db17.
// Each chunk's words are then timed by forced alignment of its text to its
// audio (`align`).
package pocket

import "base:runtime"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:strings"
import "core:time"
import "deps:oicut/common"
import "../align"
import "../assets"
import "../ort"
import "../spm"
import wav "../wav"

MIN_FRAMES_BEFORE_EOS :: 6
EOS_THRESHOLD :: -4.0
TOKENS_PER_SECOND :: 3.0
GEN_SECONDS_PADDING :: 2.0
DECODE_FRAMES :: 12
MAX_VOICE_SECONDS :: 30

// The version of what `synthesize` returns for given inputs (samples,
// words, chunks). Bumped by any change to tokenizing, chunking,
// generation, loudness or alignment that changes the output, so a store
// keyed on it (with the pins in `assets`, the text, voice and options)
// never serves a stale line.
VERSION :: 4

Options :: struct {
	temperature:      f32, // variance of the flow head's noise; upstream's default for this model is 0.3
	seed:             u64,
	lsd_steps:        int, // flow integration steps; the model is distilled for 1
	frames_after_eos: int, // < 0: upstream's guess from the prompt length
}

DEFAULT_OPTIONS :: Options {
	temperature      = 0.3,
	seed             = 0,
	lsd_steps        = 1,
	frames_after_eos = -1,
}

// Wall time per phase, summed over a synthesis.
Stats :: struct {
	chunks, frames, tokens:                     int,
	text, prompt, backbone, flow, decode, align: time.Duration,
}

// One generated chunk: its prompt text, its audio span and its words.
Chunk_Info :: struct {
	text:        string, // owned
	start, end:  f64, // seconds
	first_word:  int,
	word_count:  int,
}

Result :: struct {
	audio:       [dynamic]f32, // mono, `sample_rate`, at LOUDNESS_TARGET
	sample_rate: int,
	words:       [dynamic]Word, // every source word, in order
	chunks:      [dynamic]Chunk_Info,
	stats:       Stats,
	err:         Error, // the line's failure, from synthesize_batch
}

result_destroy :: proc(r: ^Result) {
	for c in r.chunks do delete(c.text)
	delete(r.audio)
	delete(r.words)
	delete(r.chunks)
}

Engine :: struct {
	rt:          ort.Runtime,
	text:        ort.Model,
	main:        ort.Model,
	flow:        ort.Model,
	decoder:     ort.Model,
	encoder:     ort.Model,  // loaded by the first voice_clone
	root:        string,     // the bundle's folder, for the encoder
	tok:         spm.Model,
	bundle:      Bundle,
	mimi:        Host_State, // decoder state at the start of a chunk
	bos_voice:   []f32,      // [1, 1, conditioning_dim] when insert_bos_before_voice

	main_in:     []int, // per backbone state entry: model input index
	main_out:    []int, // ...and output index
	main_seq:    int,
	main_text:   int,
	main_cond:   int,
	main_eos:    int,
	dec_in:      []int, // per Mimi state entry (-1: not a decoder input)
	dec_out:     []int,
	dec_latent:  int,
	dec_audio:   int,
	flow_c, flow_s, flow_t, flow_x: int,
	aligner:     align.Aligner,
}

@(private)
read :: proc(root, rel: string) -> ([]u8, Error) {
	path := assets.path_in(root, rel)
	data, err := os.read_entire_file(path, context.allocator)
	if err != nil do return nil, fmt.aprintf("reading %s: %v", path, err)
	return data, nil
}

@(private)
need :: proc(index: int, what: string) -> Error {
	if index < 0 do return fmt.aprintf("model has no %s", what)
	return nil
}

// Loads the bundle under `root` (as `assets.fetch` leaves it). After init
// the engine is read-only: `synthesize` may run on several threads at once
// (ONNX Runtime sessions allow concurrent runs), each with its own Result.
// ottis starts no threads and the runtime runs no graph on its own (its
// environment keeps one idle thread): every graph runs on the thread that
// calls, and `synthesize_batch` spreads lines over the caller's pool.
engine_init :: proc(e: ^Engine, root, library: string) -> Error {
	if err := ort.init(&e.rt, library); err != nil do return err
	dir :: assets.BUNDLE_NAME + "/"
	e.text = ort.load_model(&e.rt, assets.path_in(root, dir + "text_conditioner.onnx")) or_return
	e.main = ort.load_model(&e.rt, assets.path_in(root, dir + "flow_lm_main.onnx")) or_return
	e.flow = ort.load_model(&e.rt, assets.path_in(root, dir + "flow_lm_flow.onnx")) or_return
	e.decoder = ort.load_model(&e.rt, assets.path_in(root, dir + "mimi_decoder.onnx")) or_return
	e.root = strings.clone(root)

	tok_data := read(root, dir + "tokenizer.model") or_return
	defer delete(tok_data)
	tok_err: string
	e.tok, tok_err = spm.load(tok_data)
	if tok_err != "" do return tok_err

	bundle_data := read(root, dir + "bundle.json") or_return
	defer delete(bundle_data)
	e.bundle = parse_bundle(bundle_data) or_return
	if e.bundle.bundle_name != assets.BUNDLE_NAME do return fmt.aprintf("bundle is %s, expected %s", e.bundle.bundle_name, assets.BUNDLE_NAME)

	if e.bundle.insert_bos_before_voice {
		bos_data := read(root, dir + "bos_before_voice.npy") or_return
		defer delete(bos_data)
		bos_vals, bos_shape := parse_npy_f32(bos_data) or_return
		defer delete(bos_shape)
		if len(bos_vals) != e.bundle.conditioning_dim {
			delete(bos_vals)
			return fmt.aprintf("bos_before_voice.npy has %d floats, expected %d", len(bos_vals), e.bundle.conditioning_dim)
		}
		e.bos_voice = bos_vals
	}

	e.mimi = host_state_init(e.bundle.mimi_state_manifest) or_return

	// Wiring by name.
	fm := e.bundle.flow_lm_state_manifest
	e.main_in = make([]int, len(fm))
	e.main_out = make([]int, len(fm))
	for s, i in fm {
		e.main_in[i] = ort.input_index(&e.main, s.input_name)
		e.main_out[i] = ort.output_index(&e.main, s.output_name)
		need(e.main_in[i], s.input_name) or_return
		need(e.main_out[i], s.output_name) or_return
	}
	e.main_seq = ort.input_index(&e.main, "sequence")
	e.main_text = ort.input_index(&e.main, "text_embeddings")
	e.main_cond = ort.output_index(&e.main, "conditioning")
	e.main_eos = ort.output_index(&e.main, "eos_logit")
	need(e.main_seq, "sequence") or_return
	need(e.main_text, "text_embeddings") or_return
	need(e.main_cond, "conditioning") or_return
	need(e.main_eos, "eos_logit") or_return
	if len(e.main.inputs) != len(fm) + 2 do return "backbone has inputs outside the manifest"

	mm := e.bundle.mimi_state_manifest
	e.dec_in = make([]int, len(mm))
	e.dec_out = make([]int, len(mm))
	wired := 0
	for s, i in mm {
		e.dec_in[i] = ort.input_index(&e.decoder, s.input_name)
		e.dec_out[i] = ort.output_index(&e.decoder, s.output_name)
		if (e.dec_in[i] < 0) != (e.dec_out[i] < 0) do return fmt.aprintf("decoder state %s is half wired", s.input_name)
		if e.dec_in[i] >= 0 do wired += 1
	}
	e.dec_latent = ort.input_index(&e.decoder, "latent")
	need(e.dec_latent, "latent") or_return
	if wired + 1 != len(e.decoder.inputs) do return "decoder has inputs outside the manifest"
	e.dec_audio = -1
	for _, i in e.decoder.outputs {
		taken := false
		for k in e.dec_out do taken ||= k == i
		if !taken {
			if e.dec_audio >= 0 do return "decoder has more than one non-state output"
			e.dec_audio = i
		}
	}
	need(e.dec_audio, "audio output") or_return

	e.flow_c = ort.input_index(&e.flow, "c")
	e.flow_s = ort.input_index(&e.flow, "s")
	e.flow_t = ort.input_index(&e.flow, "t")
	e.flow_x = ort.input_index(&e.flow, "x")
	need(e.flow_c, "c") or_return
	need(e.flow_s, "s") or_return
	need(e.flow_t, "t") or_return
	need(e.flow_x, "x") or_return
	if len(e.flow.outputs) != 1 do return "flow head has more than one output"

	align.init(&e.aligner, &e.rt, assets.path_in(root, assets.ALIGNER.path)) or_return
	return nil
}

engine_destroy :: proc(e: ^Engine) {
	ort.unload_model(&e.text)
	ort.unload_model(&e.main)
	ort.unload_model(&e.flow)
	ort.unload_model(&e.decoder)
	ort.unload_model(&e.encoder)
	delete(e.root)
	align.destroy(&e.aligner)
	spm.destroy(&e.tok)
	host_state_destroy(&e.mimi)
	bundle_destroy(&e.bundle)
	delete(e.bos_voice)
	delete(e.main_in)
	delete(e.main_out)
	delete(e.dec_in)
	delete(e.dec_out)
	ort.destroy(&e.rt)
}

// ORT values over a host state, one per entry.
@(private)
borrow_state :: proc(e: ^Engine, s: ^Host_State, values: []^ort.Value) -> Error {
	for &v, i in values {
		v = ort.tensor(&e.rt, raw_data(s.data[i]), len(s.data[i]), s.shapes[i], s.types[i]) or_return
	}
	return nil
}

@(private)
release_all :: proc(e: ^Engine, values: []^ort.Value) {
	for &v in values {
		ort.release(&e.rt, v)
		v = nil
	}
}

// Takes each state output as the next state, releasing the old value.
@(private)
advance :: proc(e: ^Engine, state: []^ort.Value, outputs: []^ort.Value, out_index: []int) {
	for &v, i in state {
		if out_index[i] < 0 do continue
		ort.release(&e.rt, v)
		v = outputs[out_index[i]]
		outputs[out_index[i]] = nil
	}
}

// A speaker: the backbone's state after a voice prompt (either one of
// Kyutai's precomputed prompts in `assets.VOICES` or a cloned recording),
// loaded once and shared by every synthesis.
Voice :: struct {
	name:  string,
	state: Host_State,
}

// Loads a voice by name (`assets.VOICES`), from a `.safetensors` file, or
// by cloning an audio file (`.wav`, `.flac`, `.mp3`, `.ogg`).
voice_load :: proc(e: ^Engine, root, name_or_path: string) -> (v: Voice, err: Error) {
	if pin, known := assets.voice_pin(name_or_path); known {
		data := read(root, pin.path) or_return
		defer delete(data)
		return voice_from_safetensors_bytes(e, data, name_or_path)
	}
	if strings.has_suffix(strings.to_lower(name_or_path, context.temp_allocator), ".safetensors") {
		return voice_load_safetensors(e, name_or_path)
	}
	if os.exists(name_or_path) {
		return voice_clone_file(e, name_or_path)
	}
	return v, fmt.aprintf("unknown voice or missing file %q", name_or_path)
}

voice_load_safetensors :: proc(e: ^Engine, path: string, name := "") -> (v: Voice, err: Error) {
	data, rerr := os.read_entire_file(path, context.allocator)
	if rerr != nil do return v, fmt.aprintf("reading %s: %v", path, rerr)
	defer delete(data)
	label := name if name != "" else filepath.stem(path)
	return voice_from_safetensors_bytes(e, data, label)
}

@(private)
voice_from_safetensors_bytes :: proc(e: ^Engine, data: []u8, name: string) -> (v: Voice, err: Error) {
	tensors := parse_safetensors(data) or_return
	defer destroy_safetensors(&tensors)
	v.state = host_state_init(e.bundle.flow_lm_state_manifest, &tensors) or_return
	v.name = strings.clone(name)
	return v, nil
}

// Clones a voice from an audio file (WAV, FLAC, MP3, Ogg Vorbis or Opus, M4A, WebM).
voice_clone_file :: proc(e: ^Engine, path: string, name := "") -> (v: Voice, err: Error) {
	samples, rate := wav.read(path, context.temp_allocator) or_return
	label := name if name != "" else filepath.stem(path)
	return voice_clone(e, samples, rate, label)
}

// Clones a voice from mono f32 `samples` at `sample_rate` Hz: truncates to
// 30 s, resamples to 24 kHz, trims trailing silence with `end_on_pause`,
// encodes through `mimi_encoder` and runs the backbone over the voice prompt.
voice_clone :: proc(e: ^Engine, samples: []f32, sample_rate: int, name := "cloned") -> (v: Voice, err: Error) {
	if len(samples) == 0 || sample_rate <= 0 do return v, "empty voice prompt"
	if e.encoder.session == nil {
		// Only cloning needs the encoder (40 MB): speaking never loads it.
		path := assets.path_in(e.root, assets.BUNDLE_NAME + "/mimi_encoder.onnx")
		e.encoder = ort.load_model(&e.rt, path) or_return
		if len(e.encoder.inputs) != 1 || len(e.encoder.outputs) != 1 {
			ort.unload_model(&e.encoder)
			return v, "mimi_encoder: expected one input and one output"
		}
	}
	src := samples
	if max_samples := MAX_VOICE_SECONDS * sample_rate; len(src) > max_samples {
		src = src[:max_samples]
	}
	resampled := align.resample(src, sample_rate, e.bundle.sample_rate, context.temp_allocator)
	audio := end_on_pause(resampled, e.bundle.sample_rate, context.temp_allocator)
	if len(audio) < e.bundle.samples_per_frame {
		return v, "voice prompt is shorter than one frame (80 ms)"
	}

	rt := &e.rt
	audio_val := ort.tensor_f32(rt, audio, {1, 1, i64(len(audio))}) or_return
	defer ort.release(rt, audio_val)
	enc_out: [1]^ort.Value
	ort.run(&e.encoder, {audio_val}, enc_out[:]) or_return
	defer ort.release(rt, enc_out[0])

	shape_buf: [8]i64
	cond_view := ort.view(rt, enc_out[0], shape_buf[:]) or_return
	cond_floats := ort.view_f32(cond_view)
	dim := e.bundle.conditioning_dim
	if len(cond_view.shape) != 3 || int(cond_view.shape[2]) != dim {
		return v, fmt.aprintf("mimi_encoder output has unexpected shape %v", cond_view.shape)
	}
	n_cond := int(cond_view.shape[1])

	prompt_frames := n_cond + (1 if e.bundle.insert_bos_before_voice else 0)
	prompt := make([]f32, prompt_frames * dim, context.temp_allocator)
	offset := 0
	if e.bundle.insert_bos_before_voice {
		copy(prompt[:dim], e.bos_voice)
		offset = dim
	}
	copy(prompt[offset:], cond_floats)

	prompt_val := ort.tensor_f32(rt, prompt, {1, i64(prompt_frames), i64(dim)}) or_return
	defer ort.release(rt, prompt_val)
	empty_seq := ort.tensor(rt, nil, 0, {1, 0, i64(e.bundle.latent_dim)}, .Float32) or_return
	defer ort.release(rt, empty_seq)

	blank := host_state_init(e.bundle.flow_lm_state_manifest) or_return
	defer host_state_destroy(&blank)
	n_state := len(e.main_in)
	state := make([]^ort.Value, n_state, context.temp_allocator)
	borrow_state(e, &blank, state) or_return
	defer release_all(e, state)

	inputs := make([]^ort.Value, len(e.main.inputs), context.temp_allocator)
	outputs := make([]^ort.Value, len(e.main.outputs), context.temp_allocator)
	inputs[e.main_seq] = empty_seq
	inputs[e.main_text] = prompt_val
	for sv, i in state do inputs[e.main_in[i]] = sv
	ort.run(&e.main, inputs, outputs) or_return
	advance(e, state, outputs, e.main_out)
	release_all(e, outputs)

	v.state = host_state_from_values(rt, e.bundle.flow_lm_state_manifest, state) or_return
	v.name = strings.clone(name)
	return v, nil
}

// Saves a voice's backbone KV cache as a `.safetensors` file that
// `voice_load` or `voice_load_safetensors` can reload directly.
voice_save :: proc(e: ^Engine, v: ^Voice, path: string) -> Error {
	bytes := host_state_encode_safetensors(e.bundle.flow_lm_state_manifest, &v.state, context.temp_allocator)
	if err := os.write_entire_file(path, bytes); err != nil {
		return fmt.aprintf("writing %s: %v", path, err)
	}
	return nil
}

// Ends a voice prompt on `pause_sec` (80 ms) of silence after cutting
// trailing 20 ms frames more than `floor_db` (35 dB) below the loudest
// frame and fading out the last `fade_sec` (20 ms), matching upstream's
// `pocket_tts/data/audio_utils.py:end_on_pause`.
@(private)
end_on_pause :: proc(wav: []f32, sample_rate: int, allocator := context.allocator, pause_sec := 0.08, fade_sec := 0.02, floor_db: f32 = 35.0) -> []f32 {
	frame := max(1, int(0.02 * f64(sample_rate)))
	n := len(wav) / frame
	if n == 0 {
		out := make([]f32, len(wav), allocator)
		copy(out, wav)
		return out
	}
	db := make([]f32, n, context.temp_allocator)
	max_db: f32 = -1e30
	for i in 0 ..< n {
		sum: f32
		for s in wav[i * frame:][:frame] do sum += s * s
		rms := math.sqrt(sum / f32(frame))
		db[i] = 20 * math.log10(rms + 1e-12)
		if db[i] > max_db do max_db = db[i]
	}
	last_loud := 0
	for i in 0 ..< n {
		if db[i] > max_db - floor_db do last_loud = i
	}
	end := (last_loud + 1) * frame
	pause := int(pause_sec * f64(sample_rate))
	out := make([]f32, end + pause, allocator)
	copy(out[:end], wav[:end])
	fade := min(int(fade_sec * f64(sample_rate)), end)
	if fade > 0 {
		denom := f32(max(fade - 1, 1))
		for i in 0 ..< fade {
			out[end - fade + i] *= f32(fade - 1 - i) / denom
		}
	}
	return out
}

voice_destroy :: proc(v: ^Voice) {
	delete(v.name)
	host_state_destroy(&v.state)
}

// Speaks `text` in `voice` into `result` (audio, word timings, chunks),
// the line scaled to LOUDNESS_TARGET whatever the voice (loudness.odin).
// Deterministic: the same text, voice and options give the same samples.
synthesize :: proc(e: ^Engine, voice: ^Voice, text: string, options: Options, result: ^Result) -> Error {
	result.sample_rate = e.bundle.sample_rate
	chunks, ok := split_chunks(&e.tok, text, e.bundle.max_token_per_chunk, e.bundle.remove_semicolons)
	if !ok do return "nothing to say"
	source_words := word_spans(text)
	rng := rng_init(options.seed)
	word := 0
	line_start := len(result.audio)
	for chunk in chunks {
		start := f64(len(result.audio)) / f64(result.sample_rate)
		next := synthesize_chunk(e, voice, text, source_words[:], chunk, word, options, &rng, result) or_return
		end := f64(len(result.audio)) / f64(result.sample_rate)
		append(&result.chunks, Chunk_Info{strings.clone(chunk), start, end, word, next - word})
		word = next
	}
	normalize_loudness(result.audio[line_start:], result.sample_rate)
	return nil
}

// One line of a batch.
Job :: struct {
	voice:   ^Voice,
	text:    string,
	options: Options,
}

// Speaks each job into the result at its index, the lines spread over the
// caller's pool (nil: in order on the calling thread). A line runs whole on
// one thread, so its samples are the ones `synthesize` gives, at any pool
// size. A failed line leaves its error in
// its result's `err`, and the others still complete. Audio, words, chunks
// and errors come from `allocator`, which must be thread-safe when the pool
// runs ranges at once; words borrow their job's text. A line's scratch is
// the running thread's temp allocator, rewound after the line when it is
// Odin's default one.
synthesize_batch :: proc(e: ^Engine, pool: ^common.Work_Pool, jobs: []Job, results: []Result, allocator := context.allocator) {
	assert(len(results) == len(jobs))
	Batch :: struct {
		e:         ^Engine,
		jobs:      []Job,
		results:   []Result,
		allocator: mem.Allocator,
	}
	lines :: proc(data: rawptr, first, end: int) {
		b := (^Batch)(data)
		context.allocator = b.allocator
		for i in first ..< end {
			runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
			j := &b.jobs[i]
			b.results[i].err = synthesize(b.e, j.voice, j.text, j.options, &b.results[i])
		}
	}
	b := Batch{e, jobs, results, allocator}
	if pool == nil do lines(&b, 0, len(jobs))
	else do pool.run(pool, len(jobs), lines, &b)
}

@(private)
synthesize_chunk :: proc(e: ^Engine, voice: ^Voice, source: string, source_words: []Span, chunk: string, first_word: int, options: Options, rng: ^Rng, result: ^Result) -> (next_word: int, err: Error) {
	rt := &e.rt
	t0 := time.tick_now()
	prompt, guess, _ := prepare_text(chunk, e.bundle.remove_semicolons, context.temp_allocator)
	frames_after_eos := options.frames_after_eos if options.frames_after_eos >= 0 else guess + 2
	tokens := make([dynamic]spm.Token, context.temp_allocator)
	spm.encode(&e.tok, prompt, &tokens)
	next_word = first_word + len(word_spans(prompt))
	next_word = min(next_word, len(source_words))
	n_tokens := len(tokens)
	result.stats.chunks += 1
	result.stats.tokens += n_tokens

	ids := make([]i64, n_tokens, context.temp_allocator)
	for t, i in tokens do ids[i] = i64(t.id)
	ids_value := ort.tensor_i64(rt, ids, {1, i64(n_tokens)}) or_return
	defer ort.release(rt, ids_value)
	text_out: [1]^ort.Value
	if len(e.text.outputs) != 1 do return next_word, "text conditioner has more than one output"
	ort.run(&e.text, {ids_value}, text_out[:]) or_return
	embeddings := text_out[0]
	defer ort.release(rt, embeddings)
	result.stats.text += time.tick_since(t0)

	// Backbone: voice state, then the text prompt.
	t0 = time.tick_now()
	n_state := len(e.main_in)
	state := make([]^ort.Value, n_state, context.temp_allocator)
	borrow_state(e, &voice.state, state) or_return
	defer release_all(e, state)
	inputs := make([]^ort.Value, len(e.main.inputs), context.temp_allocator)
	outputs := make([]^ort.Value, len(e.main.outputs), context.temp_allocator)
	latent_dim := e.bundle.latent_dim
	empty_seq := ort.tensor(rt, nil, 0, {1, 0, i64(latent_dim)}, .Float32) or_return
	defer ort.release(rt, empty_seq)
	empty_text := ort.tensor(rt, nil, 0, {1, 0, i64(e.bundle.conditioning_dim)}, .Float32) or_return
	defer ort.release(rt, empty_text)

	run_main :: proc(e: ^Engine, seq, text: ^ort.Value, state, inputs, outputs: []^ort.Value) -> Error {
		inputs[e.main_seq] = seq
		inputs[e.main_text] = text
		for v, i in state do inputs[e.main_in[i]] = v
		ort.run(&e.main, inputs, outputs) or_return
		advance(e, state, outputs, e.main_out)
		return nil
	}
	run_main(e, empty_seq, embeddings, state, inputs, outputs) or_return
	release_all(e, outputs)
	result.stats.prompt += time.tick_since(t0)

	// Frames.
	frame_rate := e.bundle.frame_rate
	max_frames := int(math.ceil((f64(n_tokens) / TOKENS_PER_SECOND + GEN_SECONDS_PADDING) * frame_rate))
	latents := make([dynamic]f32, 0, max_frames * latent_dim, context.temp_allocator)
	current := make([]f32, latent_dim, context.temp_allocator)
	for &v in current do v = math.nan_f32()
	x := make([]f32, latent_dim, context.temp_allocator)
	std := math.sqrt(options.temperature)
	steps := max(options.lsd_steps, 1)
	dt := 1 / f32(steps)
	flow_inputs := make([]^ort.Value, len(e.flow.inputs), context.temp_allocator)
	shape_buf: [8]i64
	eos_step := -1
	for step in 0 ..< max_frames {
		t1 := time.tick_now()
		seq := ort.tensor_f32(rt, current, {1, 1, i64(latent_dim)}) or_return
		run_err := run_main(e, seq, empty_text, state, inputs, outputs)
		ort.release(rt, seq)
		if run_err != nil do return next_word, run_err
		defer release_all(e, outputs)

		eos := ort.view_f32(ort.view(rt, outputs[e.main_eos], shape_buf[:]) or_return)[0]
		if eos > EOS_THRESHOLD && eos_step < 0 && step >= MIN_FRAMES_BEFORE_EOS do eos_step = step
		if eos_step >= 0 && step >= eos_step + frames_after_eos do break

		result.stats.backbone += time.tick_since(t1)

		// Flow head: noise to latent.
		t1 = time.tick_now()
		for &v in x do v = f32(rng_normal(rng)) * std
		for j in 0 ..< steps {
			s_val := [1]f32{f32(j) * dt}
			t_val := [1]f32{f32(j) * dt + dt}
			s_value := ort.tensor_f32(rt, s_val[:], {1, 1}) or_return
			t_value := ort.tensor_f32(rt, t_val[:], {1, 1}) or_return
			x_value := ort.tensor_f32(rt, x, {1, i64(latent_dim)}) or_return
			flow_inputs[e.flow_c] = outputs[e.main_cond]
			flow_inputs[e.flow_s] = s_value
			flow_inputs[e.flow_t] = t_value
			flow_inputs[e.flow_x] = x_value
			flow_out: [1]^ort.Value
			flow_err := ort.run(&e.flow, flow_inputs, flow_out[:])
			ort.release(rt, s_value)
			ort.release(rt, t_value)
			ort.release(rt, x_value)
			if flow_err != nil do return next_word, flow_err
			direction := ort.view_f32(ort.view(rt, flow_out[0], shape_buf[:]) or_return)
			if len(direction) != latent_dim do return next_word, "flow head output has the wrong size"
			for &v, k in x do v += direction[k] * dt
			ort.release(rt, flow_out[0])
		}
		append(&latents, ..x)
		copy(current, x)
		result.stats.flow += time.tick_since(t1)
	}
	n_frames := len(latents) / latent_dim
	result.stats.frames += n_frames

	// Mimi: latents to audio, a fresh decoder state per chunk.
	t0 = time.tick_now()
	chunk_start := len(result.audio)
	dec_state := make([]^ort.Value, len(e.dec_in), context.temp_allocator)
	borrow_state(e, &e.mimi, dec_state) or_return
	defer release_all(e, dec_state)
	dec_inputs := make([]^ort.Value, len(e.decoder.inputs), context.temp_allocator)
	dec_outputs := make([]^ort.Value, len(e.decoder.outputs), context.temp_allocator)
	for first := 0; first < n_frames; first += DECODE_FRAMES {
		count := min(DECODE_FRAMES, n_frames - first)
		latent := ort.tensor_f32(rt, latents[first * latent_dim:][:count * latent_dim], {1, i64(count), i64(latent_dim)}) or_return
		dec_inputs[e.dec_latent] = latent
		for v, i in dec_state do if e.dec_in[i] >= 0 do dec_inputs[e.dec_in[i]] = v
		dec_err := ort.run(&e.decoder, dec_inputs, dec_outputs)
		ort.release(rt, latent)
		if dec_err != nil do return next_word, dec_err
		advance(e, dec_state, dec_outputs, e.dec_out)
		audio := ort.view_f32(ort.view(rt, dec_outputs[e.dec_audio], shape_buf[:]) or_return)
		if len(audio) != count * e.bundle.samples_per_frame do return next_word, fmt.aprintf("decoder returned %d samples for %d frames", len(audio), count)
		append(&result.audio, ..audio)
		release_all(e, dec_outputs)
	}
	// A fresh decoder state steps at its first samples: fade 5 ms in.
	fade := min(e.bundle.sample_rate / 200, len(result.audio) - chunk_start)
	for i in 0 ..< fade do result.audio[chunk_start + i] *= f32(i) / f32(max(e.bundle.sample_rate / 200 - 1, 1))
	result.stats.decode += time.tick_since(t0)

	// Words: the chunk's transcript aligned to its audio.
	t0 = time.tick_now()
	time_chunk_words(e, source, source_words[first_word:next_word], first_word, result.audio[chunk_start:], f64(chunk_start) / f64(e.bundle.sample_rate), &result.words)
	result.stats.align += time.tick_since(t0)
	return next_word, nil
}

// Times `words` (source spans, the first being source word `first`) in a
// chunk's audio starting `offset` seconds into the result. When the
// aligner cannot place them, they share the chunk's span by length,
// marked estimated.
@(private)
time_chunk_words :: proc(e: ^Engine, source: string, words: []Span, first: int, audio: []f32, offset: f64, out: ^[dynamic]Word) {
	n := len(words)
	texts := make([]string, n, context.temp_allocator)
	for w, k in words do texts[k] = source[w.begin:w.end]
	spans := make([]align.Span, n, context.temp_allocator)
	if err := align.align(&e.aligner, audio, e.bundle.sample_rate, texts, spans); err != nil {
		seconds := f64(len(audio)) / f64(e.bundle.sample_rate)
		total := 0
		for t in texts do total += len(t)
		at := 0.0
		for t, k in texts {
			share := seconds * f64(len(t)) / f64(max(total, 1))
			spans[k] = {at, at + share, false}
			at += share
		}
	}
	for sp, k in spans {
		append(out, Word{texts[k], first + k, offset + sp.start, offset + sp.end, !sp.aligned})
	}
}
