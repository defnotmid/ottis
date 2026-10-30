package pocket

import "core:log"
import "core:math"
import "core:os"
import "core:testing"
import "deps:oicut/common"
import "../assets"

// `seconds` of a sine at `hz` peaking at `peak_db` dBFS.
@(private = "file")
sine :: proc(hz, peak_db, seconds: f64, rate: int) -> []f32 {
	out := make([]f32, int(seconds * f64(rate)), context.temp_allocator)
	a := math.pow(10, peak_db / 20)
	for &v, i in out do v = f32(a * math.sin(2 * math.PI * hz * f64(i) / f64(rate)))
	return out
}

// A 997 Hz sine reads its peak level less 3.01 LU as one channel
// (BS.1770's reference: 0 dBFS in one channel is -3.01 LUFS), with the
// published 48 kHz filters and the derived 24 kHz ones; its true peak is its
// peak; silence and 50 ms read -inf.
@(test)
loudness_reads_reference_sine :: proc(t: ^testing.T) {
	for rate in ([]int{48000, 24000}) {
		for level in ([]f64{-20, -30}) {
			x := sine(997, level, 10, rate)
			l := integrated_loudness(x, rate)
			testing.expectf(t, abs(l - (level - 3.01)) < 0.05, "%d Hz, %.0f dBFS: %.3f LUFS", rate, level, l)
			tp := true_peak(x)
			testing.expectf(t, abs(tp - level) < 0.05, "%d Hz, %.0f dBFS: true peak %.3f", rate, level, tp)
		}
	}
	testing.expect(t, math.is_inf(integrated_loudness(make([]f32, 24000, context.temp_allocator), 24000), -1))
	testing.expect(t, math.is_inf(integrated_loudness(sine(997, -20, 0.05, 24000), 24000), -1))
}

// A quiet line comes up to the target, a loud one down, a short one too
// (one block); one whose peaks stand far over its loudness stops at the
// ceiling; silence stays silent.
@(test)
normalize_lands_on_target_under_ceiling :: proc(t: ^testing.T) {
	RATE :: 24000
	for c in ([][2]f64{{-50, 10}, {-6, 10}, {-40, 0.3}}) {
		x := sine(440, c[0], c[1], RATE)
		normalize_loudness(x, RATE)
		l := integrated_loudness(x, RATE)
		testing.expectf(t, abs(l - LOUDNESS_TARGET) < 0.01, "%.0f dBFS for %.1f s: %.3f LUFS", c[0], c[1], l)
	}

	// -50 dBFS with a full-scale click every second: 30 dB of gain would put
	// the clicks over the ceiling.
	x := sine(440, -50, 10, RATE)
	for i := RATE / 2; i < len(x); i += RATE do x[i] = 0.9
	normalize_loudness(x, RATE)
	tp := true_peak(x)
	testing.expectf(t, abs(tp - TRUE_PEAK_CEILING) < 0.01, "clicks: true peak %.3f dBTP", tp)
	testing.expect(t, integrated_loudness(x, RATE) < LOUDNESS_TARGET - 1)

	silent := make([]f32, RATE, context.temp_allocator)
	normalize_loudness(silent, RATE)
	for v in silent do testing.expect_value(t, v, 0)
}

// The same sentence in every voice on disk lands within 1 LU, under the
// ceiling: the shipped voices' raw levels span 16 LU. Needs the bundle, the
// aligner and the runtime (`ottis fetch`), and runs each voice fetched
// (`ottis fetch -voice NAME`); skipped when the model is missing.
@(test)
voices_speak_at_one_loudness :: proc(t: ^testing.T) {
	root, derr := assets.data_dir(context.temp_allocator)
	if derr != nil {
		log.infof("skipped: %s", derr.?)
		return
	}
	present := os.exists(assets.path_in(root, assets.ALIGNER.path))
	for pin in assets.BUNDLE do present &&= os.exists(assets.path_in(root, pin.path))
	if !present {
		log.infof("skipped: no model in %s (ottis fetch)", root)
		return
	}
	library, ferr := assets.fetch(root)
	if !testing.expectf(t, ferr == nil, "fetch: %v", ferr) do return
	defer delete(library)

	e: Engine
	if err := engine_init(&e, root, library); !testing.expectf(t, err == nil, "engine_init: %v", err) do return
	defer engine_destroy(&e)
	voices := make([dynamic]Voice, context.temp_allocator)
	defer for &v in voices do voice_destroy(&v)
	for pin in assets.VOICES {
		p, _ := assets.voice_pin(pin.name)
		if !os.exists(assets.path_in(root, p.path)) do continue
		v, err := voice_load(&e, root, pin.name)
		if !testing.expectf(t, err == nil, "%s: %v", pin.name, err) do return
		append(&voices, v)
	}
	if len(voices) < 2 {
		log.infof("skipped: %d voice(s) fetched", len(voices))
		return
	}

	options := DEFAULT_OPTIONS
	options.seed = 1
	jobs := make([]Job, len(voices), context.temp_allocator)
	for &v, i in voices do jobs[i] = {&v, "Each voice is a few megabytes, fetched the first time you use it.", options}
	results := make([]Result, len(jobs), context.temp_allocator)
	defer for &r in results do result_destroy(&r)
	pool: common.Static_Pool
	common.init_static_pool(&pool, clamp(os.get_processor_core_count(), 1, 8) - 1)
	synthesize_batch(&e, &pool.work_pool, jobs, results)
	common.destroy_static_pool(&pool)

	lo, hi := math.inf_f64(1), math.inf_f64(-1)
	for &r, i in results {
		if !testing.expectf(t, r.err == nil, "%s: %v", voices[i].name, r.err) do continue
		l := integrated_loudness(r.audio[:], r.sample_rate)
		tp := true_peak(r.audio[:])
		log.infof("%-15s %6.2f LUFS %6.2f dBTP", voices[i].name, l, tp)
		testing.expectf(t, tp <= TRUE_PEAK_CEILING + 0.01, "%s: true peak %.2f dBTP", voices[i].name, tp)
		lo, hi = min(lo, l), max(hi, l)
	}
	testing.expectf(t, hi - lo <= 1, "%d voices span %.2f to %.2f LUFS", len(voices), lo, hi)
}
