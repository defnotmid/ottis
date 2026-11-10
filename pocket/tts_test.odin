package pocket

import "base:runtime"
import "core:log"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import "core:testing"
import "deps:oicut/common"
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
	for c, i in heap.chunks do testing.expect_value(t, text_of(&in_temp, in_temp.chunks[i].text), text_of(&heap, c.text))
}

// A result owns what it holds: its words and chunks read the same after the
// text it was spoken from is overwritten and freed; one made in an arena is
// freed there whatever the context's allocator; and the temp allocator is
// left as it was found.
@(test)
a_result_owns_its_memory :: proc(t: ^testing.T) {
	e: Engine
	voice: Voice
	if !open_model(&e, &voice) do return
	defer engine_destroy(&e)
	defer voice_destroy(&voice)

	want: Result
	defer result_destroy(&want)
	if err := synthesize(&e, &voice, TWO_CHUNKS, DEFAULT_OPTIONS, &want); !testing.expectf(t, err == nil, "synthesize: %v", err) do return
	text := strings.clone(TWO_CHUNKS)
	temp := (^runtime.Default_Temp_Allocator)(context.temp_allocator.data)
	before := temp.arena.total_used
	got: Result
	defer result_destroy(&got)
	if err := synthesize(&e, &voice, text, DEFAULT_OPTIONS, &got); !testing.expectf(t, err == nil, "synthesize: %v", err) do return
	testing.expect_value(t, temp.arena.total_used, before)
	for &b in transmute([]u8)text do b = 'x'
	delete(text)
	if !testing.expect_value(t, len(got.words), len(want.words)) || !testing.expect_value(t, len(got.chunks), len(want.chunks)) do return
	for w, i in want.words do testing.expect_value(t, text_of(&got, got.words[i].text), text_of(&want, w.text))
	for c, i in want.chunks do testing.expect_value(t, text_of(&got, got.chunks[i].text), text_of(&want, c.text))

	arena: mem.Arena
	backing := make([]u8, 64 << 20)
	defer delete(backing)
	mem.arena_init(&arena, backing)
	jobs := []Job{{&voice, "The lamp is lit.", DEFAULT_OPTIONS}, {&voice, "", DEFAULT_OPTIONS}}
	made: [2]Result
	synthesize_batch(&e, nil, jobs, made[:], mem.arena_allocator(&arena))
	testing.expect(t, made[0].err == nil && len(made[0].words) > 0, "the line failed")
	testing.expect(t, made[1].err != nil, "an empty line did not fail")
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, context.allocator)
	defer mem.tracking_allocator_destroy(&track)
	track.bad_free_callback = mem.tracking_allocator_bad_free_callback_add_to_array
	{
		context.allocator = mem.tracking_allocator(&track)
		for &r in made do result_destroy(&r)
	}
	testing.expectf(t, len(track.bad_free_array) == 0, "%d frees of memory the context's allocator never gave out", len(track.bad_free_array))

	// A result that already holds a line keeps its allocator: a later batch
	// given another one adds to it there, its error too.
	reused: [2]Result
	synthesize_batch(&e, nil, jobs, reused[:], mem.arena_allocator(&arena))
	other: mem.Tracking_Allocator
	mem.tracking_allocator_init(&other, context.allocator)
	defer mem.tracking_allocator_destroy(&other)
	synthesize_batch(&e, nil, jobs, reused[:], mem.tracking_allocator(&other))
	testing.expect(t, reused[0].err == nil && len(reused[0].chunks) == 2, "the reused line was not added to")
	testing.expect(t, reused[1].err != nil, "an empty line did not fail")
	for &r in reused do result_destroy(&r)
	testing.expectf(t, len(other.allocation_map) == 0, "%d allocations of reused results left in the other allocator", len(other.allocation_map))
}

// A batch gives each line what `synthesize` gives it: with no pool, on a
// pool whose tasks take lines as they come, and with its results in the
// temp allocator.
@(test)
a_batch_gives_each_line_its_own_samples :: proc(t: ^testing.T) {
	e: Engine
	voice: Voice
	if !open_model(&e, &voice) do return
	defer engine_destroy(&e)
	defer voice_destroy(&voice)
	texts := []string{"The lamp is lit.", "Far below, the boats were pulling for the harbour.", "", "The gulls wheeled.", "Rain struck the glass.", "He wound the clock."}
	jobs := make([]Job, len(texts), context.temp_allocator)
	for s, i in texts do jobs[i] = {&voice, s, DEFAULT_OPTIONS}

	alone := make([]Result, len(jobs))
	pooled := make([]Result, len(jobs))
	defer {
		for &r in alone do result_destroy(&r)
		for &r in pooled do result_destroy(&r)
		delete(alone)
		delete(pooled)
	}
	for &r, i in alone do r.err = synthesize(&e, &voice, texts[i], DEFAULT_OPTIONS, &r)
	pool: common.Static_Pool
	common.init_static_pool(&pool, 2)
	synthesize_batch(&e, &pool.work_pool, jobs, pooled)
	common.destroy_static_pool(&pool)
	in_temp := make([]Result, len(jobs), context.temp_allocator)
	synthesize_batch(&e, nil, jobs, in_temp, context.temp_allocator)

	for want, i in alone {
		for got, k in ([]Result{pooled[i], in_temp[i]}) {
			how := "on the pool" if k == 0 else "in the temp allocator"
			testing.expectf(t, (got.err == nil) == (want.err == nil), "line %d %s: err %v, alone %v", i, how, got.err, want.err)
			testing.expectf(t, slice.equal(got.audio[:], want.audio[:]), "line %d %s: other samples", i, how)
			testing.expectf(t, len(got.words) == len(want.words), "line %d %s: %d words, alone %d", i, how, len(got.words), len(want.words))
		}
	}
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
