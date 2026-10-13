package align

import "core:math"
import "core:testing"

// Frames whose most likely class is given (others far less likely).
@(private = "file")
scores :: proc(best: []int) -> []f32 {
	logp := make([]f32, len(best) * 32, context.temp_allocator)
	for b, t in best {
		for k in 0 ..< 32 do logp[t * 32 + k] = -10
		logp[t * 32 + b] = -0.01
	}
	return logp
}

@(test)
viterbi_follows_letters :: proc(t: ^testing.T) {
	H, I, Y, O, U := letter_id('H'), letter_id('I'), letter_id('Y'), letter_id('O'), letter_id('U')
	// blank H H I blank | blank Y O U U blank
	best := []int{BLANK, H, H, I, BLANK, WORD_GAP, BLANK, Y, O, U, U, BLANK}
	labels := []int{H, I, WORD_GAP, Y, O, U}
	path, ok := viterbi(scores(best), len(best), labels)
	testing.expect(t, ok)
	want := []int{0, 1, 1, 3, 4, 5, 6, 7, 9, 11, 11, 12}
	for s, i in want do testing.expectf(t, path[i] == s, "frame %d: state %d, want %d", i, path[i], s)
}

@(test)
viterbi_wildcard_takes_unspelled_speech :: proc(t: ^testing.T) {
	A, B := letter_id('A'), letter_id('B')
	X, Z := letter_id('X'), letter_id('Z')
	// "a <3.50> b": the frames between say letters the transcript lacks.
	best := []int{A, BLANK, WORD_GAP, X, Z, X, BLANK, WORD_GAP, B}
	labels := []int{A, WORD_GAP, WILDCARD, WORD_GAP, B}
	path, ok := viterbi(scores(best), len(best), labels)
	testing.expect(t, ok)
	wildcard_frames := 0
	for s in path do if s == 5 do wildcard_frames += 1
	testing.expect_value(t, wildcard_frames, 3)
	testing.expect_value(t, path[0], 1)
	testing.expect_value(t, path[len(path) - 1], 9)
}

@(test)
viterbi_needs_enough_frames :: proc(t: ^testing.T) {
	A := letter_id('A')
	_, ok := viterbi(scores({A, A}), 2, {A, WORD_GAP, A, WORD_GAP, A})
	testing.expect(t, !ok)
}

@(test)
spelling_folds_and_drops :: proc(t: ^testing.T) {
	letters := make([dynamic]int, context.temp_allocator)
	spell("Café’s", &letters)
	want := []int{letter_id('C'), letter_id('A'), letter_id('F'), letter_id('E'), 27, letter_id('S')}
	testing.expect_value(t, len(letters), len(want))
	for l, i in want do testing.expect_value(t, letters[i], l)
	clear(&letters)
	spell("3.50", &letters)
	testing.expect_value(t, len(letters), 0)
}

@(test)
resample_keeps_tone_and_length :: proc(t: ^testing.T) {
	src := make([]f32, 24000, context.temp_allocator)
	for &v, i in src do v = f32(sin_440(f64(i) / 24000))
	out := resample(src, 24000, 16000, context.temp_allocator)
	testing.expect_value(t, len(out), 16000)
	worst: f64
	for v, i in out[200:len(out) - 200] {
		worst = max(worst, abs(f64(v) - sin_440(f64(i + 200) / 16000)))
	}
	testing.expectf(t, worst < 1e-3, "worst error %f", worst)
}

@(private = "file")
sin_440 :: proc(t: f64) -> f64 {
	return 0.5 * math.sin(2 * math.PI * 440 * t)
}
