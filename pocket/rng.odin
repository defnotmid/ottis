package pocket

import "core:math"

// The only randomness in generation is the flow head's starting noise, one
// Gaussian vector per frame. It comes from this generator, owned here so
// a seed keeps producing the same audio whatever Odin's math/rand becomes:
// SplitMix64 for the bits, Box-Muller for the normals.
Rng :: struct {
	state: u64,
	spare: f64,
	has_spare: bool,
}

rng_init :: proc(seed: u64) -> Rng {
	return Rng{state = seed}
}

rng_u64 :: proc(r: ^Rng) -> u64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ~ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ~ (z >> 27)) * 0x94d049bb133111eb
	return z ~ (z >> 31)
}

// Uniform in (0, 1].
@(private)
rng_unit :: proc(r: ^Rng) -> f64 {
	return f64((rng_u64(r) >> 11) + 1) * (1.0 / (1 << 53))
}

rng_normal :: proc(r: ^Rng) -> f64 {
	if r.has_spare {
		r.has_spare = false
		return r.spare
	}
	radius := math.sqrt(-2 * math.ln(rng_unit(r)))
	angle := 2 * math.PI * rng_unit(r)
	r.spare = radius * math.sin(angle)
	r.has_spare = true
	return radius * math.cos(angle)
}
