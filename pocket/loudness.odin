package pocket

import "core:math"

// Every line `synthesize` returns is scaled to one loudness, so a voice's
// level is not the recording's its prompt came from (the shipped voices
// speak from -35 to -19 LUFS): EBU R128's -23 LUFS, as a mono file (3 LU
// more when it plays on both channels of a stereo mix), by gain alone.
// Speech peaks 12 to 23 dB over its loudness, so a line whose true peak
// would pass -1 dBTP gets less gain and lands under the target by as much.
LOUDNESS_TARGET :: -23.0 // LUFS
TRUE_PEAK_CEILING :: -1.0 // dBTP

// Scales `samples` at `rate` to LOUDNESS_TARGET, or as near as the true
// peak ceiling allows. Silence stays as it is.
@(private)
normalize_loudness :: proc(samples: []f32, rate: int) {
	loudness := integrated_loudness(samples, rate)
	if math.is_inf(loudness) do return
	gain_db := min(LOUDNESS_TARGET - loudness, TRUE_PEAK_CEILING - true_peak(samples))
	gain := f32(math.pow(10, gain_db / 20))
	for &v in samples do v *= gain
}

// ITU-R BS.1770-4 integrated loudness and true peak of a mono line, measured
// as odawg's meter (and onimite's `audio analyze`) measures one channel:
// K-weighting derived at the line's rate, the K-weighted mean square per
// 100 ms step, 400 ms blocks every 100 ms gated at -70 LUFS and 10 LU under
// their mean; the true peak from a 4x Lanczos (a = 8) reconstruction.

// The integrated loudness of `samples` at `rate`, in LUFS: -inf for silence
// or less than 100 ms. A line under 400 ms is one block of its steps.
integrated_loudness :: proc(samples: []f32, rate: int) -> f64 {
	step := rate / 10
	energy := make([]f64, len(samples) / step, context.temp_allocator)
	if len(energy) == 0 do return math.inf_f64(-1)
	shelf, hp := k_coefficients(rate)
	st: [4]f64
	for &e, s in energy {
		for v in samples[s * step:][:step] do e += k_weighted_square(&st, &shelf, &hp, v)
		e /= f64(step)
	}
	width := min(4, len(energy))
	blocks := make([]f64, len(energy) - width + 1, context.temp_allocator)
	for &b, k in blocks {
		for e in energy[k:][:width] do b += e
		b /= f64(width)
	}
	// Blocks under -70 LUFS are left out, then those 10 LU under the rest's mean.
	mean_over :: proc(blocks: []f64, gate: f64) -> f64 {
		sum: f64
		count := 0
		for b in blocks do if b > gate {
			sum += b
			count += 1
		}
		return sum / f64(max(count, 1))
	}
	absolute := math.pow(10, (-70 + 0.691) / 10)
	ungated := mean_over(blocks, absolute)
	if ungated == 0 do return math.inf_f64(-1)
	return -0.691 + 10 * math.log10(mean_over(blocks, max(absolute, ungated / 10)))
}

// The true peak of `samples`, in dBTP: the peak of the signal reconstructed
// at 4x the sample rate, the three in-between phases skipped within 8
// samples of the ends. Silence reads -180.
true_peak :: proc(samples: []f32) -> f64 {
	TAPS :: 8
	kernel: [3][2 * TAPS]f32
	for &row, ph in kernel {
		frac := f64(ph + 1) / 4
		for &w, k in row do w = f32(lanczos(f64(k - TAPS + 1) - frac, TAPS))
	}
	peak: f32
	for v in samples do peak = max(peak, abs(v))
	for i in TAPS - 1 ..< len(samples) - TAPS {
		win := samples[i - TAPS + 1:][:2 * TAPS]
		for row in kernel {
			sum: f32
			for k in 0 ..< 2 * TAPS do sum += win[k] * row[k]
			peak = max(peak, abs(sum))
		}
	}
	return 20 * math.log10(max(f64(peak), 1e-9))
}

@(private)
K_Biquad :: struct {
	b0, b1, b2, a1, a2: f64,
}

// BS.1770's K-weighting filters at `rate`: the published 48 kHz
// coefficients, or at other rates the same analog prototypes through the
// bilinear transform (libebur128's derivation, as odawg's).
@(private)
k_coefficients :: proc(rate: int) -> (shelf, hp: K_Biquad) {
	if rate == 48000 {
		return {1.53512485958697, -2.69169618940638, 1.19839281085285, -1.69065929318241, 0.73248077421585},
			{1, -2, 1, -1.99004745483398, 0.99007225036621}
	}
	fs := f64(rate)
	{
		f0 := 1681.974450955533
		g := 3.999843853973347
		q := 0.7071752369554196
		k := math.tan(math.PI * f0 / fs)
		vh := math.pow(10, g / 20)
		vb := math.pow(vh, 0.4996667741545416)
		a0 := 1 + k / q + k * k
		shelf = {(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0}
	}
	{
		f0 := 38.13547087602444
		q := 0.5003270373238773
		k := math.tan(math.PI * f0 / fs)
		a0 := 1 + k / q + k * k
		hp = {1, -2, 1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0}
	}
	return
}

// A sample through the K-weighting (high shelf, then RLB high-pass, direct
// form II transposed), squared.
@(private)
k_weighted_square :: #force_inline proc "contextless" (st: ^[4]f64, shelf, hp: ^K_Biquad, v: f32) -> f64 {
	fv := f64(v)
	y := shelf.b0 * fv + st[0]
	st[0] = shelf.b1 * fv - shelf.a1 * y + st[1]
	st[1] = shelf.b2 * fv - shelf.a2 * y
	z := hp.b0 * y + st[2]
	st[2] = hp.b1 * y - hp.a1 * z + st[3]
	st[3] = hp.b2 * y - hp.a2 * z
	w := f64(f32(z))
	return w * w
}

@(private)
lanczos :: proc(x, a: f64) -> f64 {
	if x == 0 do return 1
	if abs(x) >= a do return 0
	px := math.PI * x
	return a * math.sin(px) * math.sin(px / a) / (px * px)
}
