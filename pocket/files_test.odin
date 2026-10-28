package pocket

import "core:mem"
import "core:testing"

// An .npy file with `header` and `body` zero bytes after it.
@(private = "file")
npy :: proc(version: u8, header: string, body: int) -> []u8 {
	buf := make([dynamic]u8, context.temp_allocator)
	append(&buf, "\x93NUMPY")
	append(&buf, version, 0)
	n := len(header)
	append(&buf, u8(n), u8(n >> 8))
	if version > 1 do append(&buf, u8(n >> 16), u8(n >> 24))
	append(&buf, header)
	for _ in 0 ..< body do append(&buf, 0)
	return buf[:]
}

// Files cut short or lying about their contents are refused (not read past
// their end) and leave nothing allocated: a version 2 .npy too short for
// its header length, a shape with no ')', negative dimensions, dimensions
// whose product overflows, a body the shape does not match; a .safetensors
// header length past the file, a negative dimension after a good tensor.
@(test)
files_refuse_bad_input :: proc(t: ^testing.T) {
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, context.allocator)
	defer mem.tracking_allocator_destroy(&track)
	context.allocator = mem.tracking_allocator(&track)

	Case :: struct {
		data: []u8,
		err:  string,
	}
	F32 :: "{'descr': '<f4', 'fortran_order': False, "
	cases := []Case {
		{transmute([]u8)string("\x93NUMPY\x02\x00\x10\x00"), "npy: truncated header"},
		{npy(1, F32 + "'shape': (2, }", 8), "npy: bad shape"},
		{npy(1, F32 + "'shape': (-2, -3), }", 24), "npy: bad shape"},
		{npy(2, F32 + "'shape': (4611686018427387904, 4), }", 16), "npy: size does not match shape"},
		{npy(1, F32 + "'shape': (3,), }", 8), "npy: size does not match shape"},
	}
	for c in cases {
		values, shape, err := parse_npy_f32(c.data)
		testing.expectf(t, err.? == c.err, "%q: %v, want %s", c.data, err, c.err)
		testing.expect(t, values == nil && shape == nil)
	}
	values, shape, err := parse_npy_f32(npy(1, F32 + "'shape': (1, 2), }", 8))
	testing.expect(t, err == nil && len(values) == 2 && len(shape) == 2 && shape[0] == 1 && shape[1] == 2)
	delete(values)
	delete(shape)

	huge := []u8{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, '{', '}'}
	tensors, terr := parse_safetensors(huge)
	testing.expect_value(t, terr.?, "safetensors: truncated header")
	testing.expect(t, tensors == nil)

	header := `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}`
	file := make([dynamic]u8, context.temp_allocator)
	append(&file, u8(len(header)), 0, 0, 0, 0, 0, 0, 0)
	append(&file, header)
	append(&file, 0, 0, 0, 0)
	tensors, terr = parse_safetensors(file[:])
	testing.expect_value(t, terr.?, "safetensors: bad shape for b")
	testing.expect(t, tensors == nil)
	delete(terr.?)

	testing.expectf(t, len(track.allocation_map) == 0, "%d allocations left", len(track.allocation_map))
}

// A saved tensor whose data is not as long as its shape says is refused
// before anything is copied.
@(test)
overlay_checks_saved_tensor :: proc(t: ^testing.T) {
	dst := make([]u8, 4 * 4 * 4, context.temp_allocator)
	src := Safetensor{dtype = "F32", shape = {2, 2}, data = make([]u8, 8, context.temp_allocator)}
	err := overlay(dst, .Float32, {4, 4}, src)
	testing.expect_value(t, err.?, "voice tensor's data does not match its shape")
	src.data = make([]u8, 16, context.temp_allocator)
	for &b in src.data do b = 1
	testing.expect(t, overlay(dst, .Float32, {4, 4}, src) == nil)
	testing.expect(t, dst[0] == 1 && dst[7] == 1 && dst[8] == 0 && dst[16] == 1)
}
