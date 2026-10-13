package align

import "core:math"

// Windowed-sinc (Lanczos, a = 8) resampling of mono samples, low-passed
// at the lower of the two Nyquist rates. Lengths scale exactly:
// len(out) = floor(len(src) * to / from).
resample :: proc(src: []f32, from_rate, to_rate: int, allocator := context.allocator) -> []f32 {
	if from_rate == to_rate {
		out := make([]f32, len(src), allocator)
		copy(out, src)
		return out
	}
	n := len(src) * to_rate / from_rate
	out := make([]f32, n, allocator)
	ratio := f64(from_rate) / f64(to_rate)
	cutoff := min(1, 1 / ratio) // of the source Nyquist
	A :: 8.0
	half := A / cutoff
	for i in 0 ..< n {
		center := f64(i) * ratio
		lo := max(0, int(math.ceil(center - half)))
		hi := min(len(src) - 1, int(math.floor(center + half)))
		acc, norm: f64
		for j in lo ..= hi {
			x := (f64(j) - center) * cutoff
			w := lanczos(x, A) * cutoff
			acc += f64(src[j]) * w
			norm += w
		}
		out[i] = f32(acc / norm) if norm != 0 else 0
	}
	return out
}

@(private)
lanczos :: proc(x, a: f64) -> f64 {
	if x == 0 do return 1
	if abs(x) >= a do return 0
	px := math.PI * x
	return a * math.sin(px) * math.sin(px / a) / (px * px)
}
