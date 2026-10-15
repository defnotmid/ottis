package pocket

import "core:encoding/endian"
import "core:fmt"
import "core:math"
import "core:mem"
import "../ort"

// A graph's recurrent state in host memory: the start of every chunk.
// Runs borrow it through ORT values and get their next state back as
// ORT-allocated outputs, so it is never written after it is built.
Host_State :: struct {
	types:  []ort.Element_Type,
	shapes: [][]i64,
	data:   [][]u8,
}

@(private)
element :: proc(dtype: string) -> (ort.Element_Type, int, bool) {
	switch dtype {
	case "float32", "F32": return .Float32, 4, true
	case "int64", "I64":   return .Int64, 8, true
	case "bool", "BOOL":   return .Bool, 1, true
	}
	return .Undefined, 0, false
}

// The state a manifest declares, filled as declared ("nan", "ones",
// "zeros", "empty"), then overlaid with a saved voice's tensors where it
// has them (`module/key`; a KV cache shorter than the graph's is copied
// into its leading positions, and `step` comes from the saved `offset`).
host_state_init :: proc(manifest: []State_Entry, voice: ^map[string]Safetensor = nil) -> (s: Host_State, err: Error) {
	n := len(manifest)
	s.types = make([]ort.Element_Type, n)
	s.shapes = make([][]i64, n)
	s.data = make([][]u8, n)
	for e, i in manifest {
		type, width, ok := element(e.dtype)
		if !ok do return s, fmt.aprintf("state %s: unsupported dtype %s", e.input_name, e.dtype)
		count := 1
		for d in e.shape do count *= int(d)
		s.types[i] = type
		s.shapes[i] = e.shape
		s.data[i] = make([]u8, count * width)
		fill(s.data[i], type, e.fill)

		if voice == nil do continue
		key := fmt.tprintf("%s/%s", e.module, e.key)
		src, found := voice[key]
		if !found && e.key == "step" {
			src, found = voice[fmt.tprintf("%s/offset", e.module)]
		}
		if !found do continue
		overlay(s.data[i], type, e.shape, src) or_return
	}
	return s, nil
}

host_state_destroy :: proc(s: ^Host_State) {
	for d in s.data do delete(d)
	delete(s.types)
	delete(s.shapes)
	delete(s.data)
}

@(private)
fill :: proc(data: []u8, type: ort.Element_Type, how: string) {
	switch how {
	case "nan":
		if type == .Float32 {
			values := mem.slice_data_cast([]f32, data)
			for &v in values do v = math.nan_f32()
		}
	case "ones":
		switch type {
		case .Float32: for &v in mem.slice_data_cast([]f32, data) do v = 1
		case .Int64: for &v in mem.slice_data_cast([]i64, data) do v = 1
		case .Bool, .Uint8, .Int8: for &v in data do v = 1
		case .Undefined, .Float16:
		}
	}
}

// Copies a saved tensor into a state tensor of the same type: whole when
// the element counts agree, else the leading block of each dimension.
@(private)
overlay :: proc(dst: []u8, type: ort.Element_Type, dst_shape: []i64, src: Safetensor) -> Error {
	src_type, width, ok := element(src.dtype)
	if !ok || src_type != type do return fmt.aprintf("voice tensor type %s does not match state %v", src.dtype, type)
	if len(src.data) == len(dst) {
		copy(dst, src.data)
		return nil
	}
	if len(src.shape) != len(dst_shape) do return fmt.aprintf("voice tensor rank %d does not match state rank %d", len(src.shape), len(dst_shape))
	for d, i in src.shape do if d > dst_shape[i] do return "voice tensor is larger than the state"
	copy_block(dst, dst_shape, src.data, src.shape, width)
	return nil
}

@(private)
copy_block :: proc(dst: []u8, dst_shape: []i64, src: []u8, src_shape: []i64, width: int) {
	if len(src_shape) == 1 {
		copy(dst, src[:int(src_shape[0]) * width])
		return
	}
	dst_stride, src_stride := width, width
	for d in dst_shape[1:] do dst_stride *= int(d)
	for d in src_shape[1:] do src_stride *= int(d)
	for i in 0 ..< int(src_shape[0]) {
		copy_block(dst[i * dst_stride:], dst_shape[1:], src[i * src_stride:], src_shape[1:], width)
	}
}

// The value of an int64 state tensor (a layer's position).
host_i64 :: proc(s: ^Host_State, i: int) -> i64 {
	assert(s.types[i] == .Int64 && len(s.data[i]) >= 8)
	return i64(endian.unchecked_get_u64le(s.data[i]))
}

// Copies ORT state values into a new Host_State matching `manifest`.
host_state_from_values :: proc(rt: ^ort.Runtime, manifest: []State_Entry, values: []^ort.Value) -> (s: Host_State, err: Error) {
	n := len(manifest)
	s.types = make([]ort.Element_Type, n)
	s.shapes = make([][]i64, n)
	s.data = make([][]u8, n)
	shape_buf: [8]i64
	for e, i in manifest {
		type, width, ok := element(e.dtype)
		if !ok do return s, fmt.aprintf("state %s: unsupported dtype %s", e.input_name, e.dtype)
		t := ort.view(rt, values[i], shape_buf[:]) or_return
		if t.type != type do return s, fmt.aprintf("state %s: got %v, expected %v", e.input_name, t.type, type)
		s.types[i] = type
		s.shapes[i] = e.shape
		bytes := t.count * width
		s.data[i] = make([]u8, bytes)
		if bytes > 0 && t.data != nil {
			copy(s.data[i], ([^]u8)(t.data)[:bytes])
		}
	}
	return s, nil
}

// Serializes a backbone voice state to .safetensors (slicing KV caches to
// the voice prompt's `step` positions, matching upstream's exported voices).
host_state_encode_safetensors :: proc(manifest: []State_Entry, s: ^Host_State, allocator := context.allocator) -> []u8 {
	step: i64 = 0
	for e, i in manifest {
		if e.key == "step" && s.types[i] == .Int64 && len(s.data[i]) >= 8 {
			step = host_i64(s, i)
			break
		}
	}
	entries := make([dynamic]Named_Safetensor, 0, len(manifest), context.temp_allocator)
	for e, i in manifest {
		switch e.key {
		case "cache":
			// [2, 1, 1000, 16, 64] -> slice dim 2 to `step`
			if len(e.shape) != 5 || step <= 0 || step > e.shape[2] do continue
			sliced_shape := make([]i64, 5, context.temp_allocator)
			copy(sliced_shape, e.shape)
			sliced_shape[2] = step
			_, width, _ := element(e.dtype)
			row_bytes := int(e.shape[3] * e.shape[4]) * width
			full_stride := int(e.shape[2]) * row_bytes
			sliced_stride := int(step) * row_bytes
			outer := int(e.shape[0] * e.shape[1])
			buf := make([]u8, outer * sliced_stride, context.temp_allocator)
			for k in 0 ..< outer {
				copy(buf[k * sliced_stride:][:sliced_stride], s.data[i][k * full_stride:][:sliced_stride])
			}
			append(&entries, Named_Safetensor{
				name   = fmt.tprintf("%s/cache", e.module),
				tensor = {dtype = "F32", shape = sliced_shape, data = buf},
			})
		case "step":
			append(&entries, Named_Safetensor{
				name   = fmt.tprintf("%s/offset", e.module),
				tensor = {dtype = "I64", shape = e.shape, data = s.data[i]},
			})
		}
	}
	return encode_safetensors(entries[:], allocator)
}

