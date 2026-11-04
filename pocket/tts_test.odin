package pocket

import "core:log"
import "core:os"
import "core:slice"
import "core:testing"
import "../assets"

@(private = "file")
TWO_CHUNKS :: "The keeper climbed the spiral stairs, forty two of them, and lit the lamp before the storm came in from the west. Far below, the boats were pulling for the harbour while the gulls wheeled over the breakwater and the first rain struck the glass."

// The engine and a voice from the data directory; false when they are
// missing (`ottis fetch`), and the test skips.
@(private = "file")
open_model :: proc(e: ^Engine, voice: ^Voice) -> bool {
	root, derr := assets.data_dir(context.temp_allocator)
	if derr != nil {
		log.infof("skipped: %s", derr.?)
		delete(derr.?)
		return false
	}
	pin, _ := assets.voice_pin("alba")
	present := os.exists(assets.path_in(root, assets.ALIGNER.path)) && os.exists(assets.path_in(root, pin.path))
	for p in assets.BUNDLE do present &&= os.exists(assets.path_in(root, p.path))
	if !present {
		log.infof("skipped: no model in %s (ottis fetch)", root)
		return false
	}
	library, ferr := assets.fetch(root)
	defer delete(library)
	if ferr != nil {
		log.errorf("fetch: %s", ferr.?)
		delete(ferr.?)
		return false
	}
	if err := engine_init(e, root, library); err != nil {
		log.errorf("engine_init: %s", err.?)
		delete(err.?)
		return false
	}
	verr: Error
	voice^, verr = voice_load(e, root, "alba")
	if verr != nil {
		log.errorf("voice_load: %s", verr.?)
		delete(verr.?)
		engine_destroy(e)
		return false
	}
	return true
}

// A result stays in the caller's allocator, whichever it is: nothing rewinds
// the temp allocator under it.
@(test)
a_result_stays_in_the_callers_allocator :: proc(t: ^testing.T) {
	e: Engine
	voice: Voice
	if !open_model(&e, &voice) do return
	defer engine_destroy(&e)
	defer voice_destroy(&voice)
	options := DEFAULT_OPTIONS
	options.seed = 1

	heap: Result
	defer result_destroy(&heap)
	err := synthesize(&e, &voice, TWO_CHUNKS, options, &heap)
	if !testing.expectf(t, err == nil, "synthesize: %v", err) do return
	testing.expectf(t, len(heap.chunks) >= 2, "%d chunks", len(heap.chunks))

	in_temp: Result
	{
		context.allocator = context.temp_allocator
		terr := synthesize(&e, &voice, TWO_CHUNKS, options, &in_temp)
		if !testing.expectf(t, terr == nil, "synthesize: %v", terr) do return
	}
	testing.expect(t, slice.equal(heap.audio[:], in_temp.audio[:]), "samples in the temp allocator differ")
	testing.expect_value(t, len(in_temp.chunks), len(heap.chunks))
	for c, i in heap.chunks do testing.expect_value(t, in_temp.chunks[i].text, c.text)
}

// A voice so far along the KV cache that a line cannot fit fails: it does not
// return the part that fit.
@(test)
a_line_past_the_kv_cache_fails :: proc(t: ^testing.T) {
	e: Engine
	voice: Voice
	if !open_model(&e, &voice) do return
	defer engine_destroy(&e)
	defer voice_destroy(&voice)
	for entry, i in e.bundle.flow_lm_state_manifest do if entry.key == "step" {
		(^i64)(raw_data(voice.state.data[i]))^ = 949
	}
	result: Result
	defer result_destroy(&result)
	err := synthesize(&e, &voice, TWO_CHUNKS, DEFAULT_OPTIONS, &result)
	testing.expect(t, err != nil, "synthesize returned no error")
	if err != nil do delete(err.?)
}
