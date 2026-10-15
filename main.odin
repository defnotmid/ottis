// ottis: text to speech with Kyutai's Pocket TTS, and word timings, in
// one native binary. It downloads its pinned runtime and model on first
// use (`ottis fetch` does it up front).
package main

import "core:crypto/sha2"
import "core:encoding/hex"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"
import "assets"
import "pocket"
import wav "wav"

USAGE :: `usage:
  ottis say TEXT [-o out.wav] [-words out.json] [-voice NAME|FILE] [-seed N]
                 [-temperature T]
  ottis say -f FILE ...        text from a file ("-" for stdin)
  ottis clone AUDIO [-o out.safetensors]
  ottis fetch [-voice NAME] [-verify]
  ottis voices
common: -data DIR (default $OTTIS_DATA, else the user data directory)`

Args :: struct {
	command, text, text_file, out, words, voice, data: string,
	options:                                           pocket.Options,
	verify:                                            bool,
}

fail :: proc(format: string, args: ..any) -> ! {
	fmt.eprintfln(format, ..args)
	os.exit(1)
}

parse_args :: proc() -> Args {
	a := Args{voice = assets.DEFAULT_VOICE, options = pocket.DEFAULT_OPTIONS}
	argv := os.args[1:]
	if len(argv) == 0 do fail(USAGE)
	a.command = argv[0]
	value :: proc(argv: []string, i: ^int) -> string {
		i^ += 1
		if i^ >= len(argv) do fail("%s needs a value", argv[i^ - 1])
		return argv[i^]
	}
	for i := 1; i < len(argv); i += 1 {
		switch argv[i] {
		case "-o": a.out = value(argv, &i)
		case "-words": a.words = value(argv, &i)
		case "-voice": a.voice = value(argv, &i)
		case "-data": a.data = value(argv, &i)
		case "-f": a.text_file = value(argv, &i)
		case "-verify": a.verify = true
		case "-seed":
			v, ok := strconv.parse_u64(value(argv, &i))
			if !ok do fail("bad -seed")
			a.options.seed = v
		case "-temperature":
			v, ok := strconv.parse_f32(value(argv, &i))
			if !ok || v < 0 do fail("bad -temperature")
			a.options.temperature = v
		case:
			if strings.has_prefix(argv[i], "-") || a.text != "" do fail("unexpected %q\n%s", argv[i], USAGE)
			a.text = argv[i]
		}
	}
	return a
}

progress :: proc(message: string) {
	fmt.eprintln(message)
}

main :: proc() {
	a := parse_args()
	root := a.data
	if root == "" {
		dir, err := assets.data_dir()
		if err != nil do fail("%s", err.?)
		root = dir
	}

	switch a.command {
	case "voices":
		for v in assets.VOICES do fmt.println(v.name)
	case "fetch":
		library, err := assets.fetch(root, a.voice, a.verify, progress)
		if err != nil do fail("%s", err.?)
		fmt.printfln("ready in %s (%s)", root, library)
	case "clone":
		clone(a, root)
	case "say":
		say(a, root)
	case:
		fail(USAGE)
	}
}

clone :: proc(a: Args, root: string) {
	if a.text == "" do fail("missing audio file\n%s", USAGE)
	out := a.out if a.out != "" else "voice.safetensors"

	library, ferr := assets.fetch(root, "", false, progress)
	if ferr != nil do fail("%s", ferr.?)

	engine: pocket.Engine
	if err := pocket.engine_init(&engine, root, library); err != nil do fail("%s", err.?)
	defer pocket.engine_destroy(&engine)

	start := time.tick_now()
	voice, verr := pocket.voice_clone_file(&engine, a.text)
	if verr != nil do fail("%s", verr.?)
	defer pocket.voice_destroy(&voice)
	elapsed := time.tick_since(start)

	if err := pocket.voice_save(&engine, &voice, out); err != nil do fail("%s", err.?)
	fmt.printfln("%s: cloned %s in %.0f ms", out, a.text, time.duration_milliseconds(elapsed))
}

say :: proc(a: Args, root: string) {
	out := a.out if a.out != "" else "out.wav"
	text := a.text
	if a.text_file != "" {
		data: []u8
		err: os.Error
		if a.text_file == "-" {
			data, err = os.read_entire_file(os.stdin, context.allocator)
		} else {
			data, err = os.read_entire_file(a.text_file, context.allocator)
		}
		if err != nil do fail("reading %s: %v", a.text_file, err)
		text = string(data)
	}
	if strings.trim_space(text) == "" do fail("nothing to say\n%s", USAGE)

	library, ferr := assets.fetch(root, a.voice, false, progress)
	if ferr != nil do fail("%s", ferr.?)

	load_start := time.tick_now()
	engine: pocket.Engine
	if err := pocket.engine_init(&engine, root, library); err != nil do fail("%s", err.?)
	defer pocket.engine_destroy(&engine)
	voice, verr := pocket.voice_load(&engine, root, a.voice)
	if verr != nil do fail("%s", verr.?)
	defer pocket.voice_destroy(&voice)
	load := time.tick_since(load_start)

	start := time.tick_now()
	result: pocket.Result
	defer pocket.result_destroy(&result)
	if err := pocket.synthesize(&engine, &voice, text, a.options, &result); err != nil do fail("%s", err.?)
	wall := time.tick_since(start)

	if err := wav.write_pcm16(out, result.audio[:], result.sample_rate); err != nil do fail("writing %s: %v", out, err)
	if a.words != "" {
		if err := os.write_entire_file(a.words, words_json(text, &result)); err != nil do fail("writing %s: %v", a.words, err)
	}

	seconds := f64(len(result.audio)) / f64(result.sample_rate)
	s := result.stats
	ms :: proc(d: time.Duration) -> f64 {return time.duration_milliseconds(d)}
	fmt.printfln("%s: %.2f s of audio, %d chunks, %d tokens, %d frames, voice %s, seed %d", out, seconds, s.chunks, s.tokens, s.frames, voice.name, a.options.seed)
	fmt.printfln("load %.0f ms; synthesis %.0f ms (%.2fx real time): text %.0f, prompt %.0f, backbone %.0f, flow %.0f, decode %.0f, align %.1f ms", ms(load), ms(wall), seconds / time.duration_seconds(wall), ms(s.text), ms(s.prompt), ms(s.backbone), ms(s.flow), ms(s.decode), ms(s.align))
	fmt.printfln("samples sha256 %s", samples_hash(result.audio[:]))
}

samples_hash :: proc(samples: []f32) -> string {
	ctx: sha2.Context_256
	sha2.init_256(&ctx)
	sha2.update(&ctx, mem.slice_to_bytes(samples))
	digest: [sha2.DIGEST_SIZE_256]u8
	sha2.final(&ctx, digest[:])
	return string(hex.encode(digest[:], context.temp_allocator))
}

// OpenAI Whisper's verbose_json shape, which onimite's captions{} reads: a
// segment per generated chunk with its words, times in seconds; a word
// whose time is an estimate (not aligned letter by letter) says so.
words_json :: proc(text: string, r: ^pocket.Result) -> string {
	b := strings.builder_make()
	quote :: proc(b: ^strings.Builder, s: string) {
		strings.write_byte(b, '"')
		for c in s {
			switch c {
			case '"': strings.write_string(b, `\"`)
			case '\\': strings.write_string(b, `\\`)
			case '\n': strings.write_string(b, `\n`)
			case '\r': strings.write_string(b, `\r`)
			case '\t': strings.write_string(b, `\t`)
			case:
				if c < 0x20 {
					fmt.sbprintf(b, `\u%04x`, int(c))
				} else {
					strings.write_rune(b, c)
				}
			}
		}
		strings.write_byte(b, '"')
	}
	fmt.sbprintf(&b, `{{"text": `)
	quote(&b, text)
	fmt.sbprintf(&b, `, "duration": %.3f, "segments": [`, f64(len(r.audio)) / f64(r.sample_rate))
	for c, i in r.chunks {
		if i > 0 do strings.write_string(&b, ",")
		fmt.sbprintf(&b, "\n  {{\"id\": %d, \"start\": %.3f, \"end\": %.3f, \"text\": ", i, c.start, c.end)
		quote(&b, c.text)
		strings.write_string(&b, `, "words": [`)
		for w, k in r.words[c.first_word:][:c.word_count] {
			if k > 0 do strings.write_string(&b, ",")
			strings.write_string(&b, "\n    {\"word\": ")
			quote(&b, w.text)
			fmt.sbprintf(&b, `, "start": %.3f, "end": %.3f`, w.start, w.end)
			if w.estimated do strings.write_string(&b, `, "estimated": true`)
			strings.write_string(&b, "}")
		}
		strings.write_string(&b, "]}")
	}
	strings.write_string(&b, "\n]}\n")
	return strings.to_string(b)
}
