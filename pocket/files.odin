package pocket

import "core:encoding/endian"
import "core:encoding/json"
import "core:fmt"
import "core:strconv"
import "core:strings"

Error :: Maybe(string)

// One recurrent tensor a graph takes as `input_name` and returns as
// `output_name` (bundle.json's state manifests).
State_Entry :: struct {
	input_name:  string,
	output_name: string,
	module:      string,
	key:         string,
	dtype:       string,
	fill:        string,
	shape:       []i64,
	index:       int,
}

Bundle :: struct {
	bundle_name:               string,
	sample_rate:               int,
	samples_per_frame:         int,
	frame_rate:                f64,
	latent_dim:                int,
	conditioning_dim:          int,
	max_token_per_chunk:       int,
	insert_bos_before_voice:   bool,
	remove_semicolons:         bool,
	pad_with_spaces_for_short_inputs: bool,
	flow_lm_state_manifest:    []State_Entry,
	mimi_state_manifest:       []State_Entry,
}

parse_bundle :: proc(data: []u8, allocator := context.allocator) -> (b: Bundle, err: Error) {
	if jerr := json.unmarshal(data, &b, allocator = allocator); jerr != nil {
		return b, fmt.aprintf("bundle.json: %v", jerr)
	}
	if b.sample_rate <= 0 || b.samples_per_frame <= 0 || b.latent_dim <= 0 || b.conditioning_dim <= 0 {
		return b, "bundle.json: missing dimensions"
	}
	return b, nil
}

// A float32 .npy array (C order).
parse_npy_f32 :: proc(data: []u8) -> (values: []f32, shape: [dynamic]i64, err: Error) {
	if len(data) < 10 || string(data[:6]) != "\x93NUMPY" do return nil, shape, "npy: bad magic"
	header_len, start: int
	switch data[6] {
	case 1:
		header_len, start = int(endian.unchecked_get_u16le(data[8:])), 10
	case 2, 3:
		header_len, start = int(endian.unchecked_get_u32le(data[8:])), 12
	case:
		return nil, shape, "npy: unknown version"
	}
	if start + header_len > len(data) do return nil, shape, "npy: truncated header"
	header := string(data[start:][:header_len])
	if !strings.contains(header, "'descr': '<f4'") do return nil, shape, fmt.aprintf("npy: not little-endian float32: %s", header)
	if strings.contains(header, "'fortran_order': True") do return nil, shape, "npy: Fortran order"
	open := strings.index(header, "'shape': (")
	if open < 0 do return nil, shape, "npy: no shape"
	dims := header[open + len("'shape': ("):]
	dims = dims[:strings.index_byte(dims, ')')]
	count := 1
	for part in strings.split_iterator(&dims, ",") {
		p := strings.trim_space(part)
		if p == "" do continue
		v, ok := strconv.parse_i64(p)
		if !ok do return nil, shape, "npy: bad shape"
		append(&shape, v)
		count *= int(v)
	}
	body := data[start + header_len:]
	if len(body) != count * 4 do return nil, shape, "npy: size does not match shape"
	values = make([]f32, count)
	for &v, i in values do v = transmute(f32)endian.unchecked_get_u32le(body[i * 4:])
	return values, shape, nil
}

Safetensor :: struct {
	dtype: string,
	shape: []i64,
	data:  []u8,
}

// The tensors of a .safetensors file, borrowing `data`.
parse_safetensors :: proc(data: []u8, allocator := context.allocator) -> (tensors: map[string]Safetensor, err: Error) {
	context.allocator = allocator
	if len(data) < 8 do return tensors, "safetensors: truncated"
	n := int(endian.unchecked_get_u64le(data))
	if 8 + n > len(data) do return tensors, "safetensors: truncated header"
	body := data[8 + n:]
	root, perr := json.parse(data[8:][:n], parse_integers = true, allocator = context.temp_allocator)
	if perr != nil do return tensors, fmt.aprintf("safetensors: %v", perr)
	object, is_object := root.(json.Object)
	if !is_object do return tensors, "safetensors: header is not an object"
	for name, value in object {
		if name == "__metadata__" do continue
		entry, ok := value.(json.Object)
		if !ok do return tensors, fmt.aprintf("safetensors: bad entry %s", name)
		t: Safetensor
		dtype, _ := entry["dtype"].(json.String)
		t.dtype = strings.clone(dtype)
		shape, _ := entry["shape"].(json.Array)
		t.shape = make([]i64, len(shape))
		for d, i in shape {
			v, is_int := d.(json.Integer)
			if !is_int do return tensors, fmt.aprintf("safetensors: bad shape for %s", name)
			t.shape[i] = v
		}
		offsets, _ := entry["data_offsets"].(json.Array)
		if len(offsets) != 2 do return tensors, fmt.aprintf("safetensors: bad offsets for %s", name)
		a, _ := offsets[0].(json.Integer)
		b, _ := offsets[1].(json.Integer)
		if a < 0 || b < a || int(b) > len(body) do return tensors, fmt.aprintf("safetensors: offsets out of range for %s", name)
		t.data = body[a:b]
		tensors[strings.clone(name)] = t
	}
	return tensors, nil
}

destroy_safetensors :: proc(tensors: ^map[string]Safetensor) {
	for name, t in tensors {
		delete(name)
		delete(t.dtype)
		delete(t.shape)
	}
	delete(tensors^)
}

Named_Safetensor :: struct {
	name:   string,
	tensor: Safetensor,
}

// Encodes `entries` in .safetensors format (8-byte little-endian header size,
// JSON header with 0-based byte offsets into the body, then concatenated data).
encode_safetensors :: proc(entries: []Named_Safetensor, allocator := context.allocator) -> []u8 {
	b := strings.builder_make(context.temp_allocator)
	strings.write_byte(&b, '{')
	offset := 0
	for e, i in entries {
		if i > 0 do strings.write_byte(&b, ',')
		fmt.sbprintf(&b, "%q:{{\"dtype\":%q,\"shape\":[", e.name, e.tensor.dtype)
		for d, k in e.tensor.shape {
			if k > 0 do strings.write_byte(&b, ',')
			fmt.sbprintf(&b, "%d", d)
		}
		next := offset + len(e.tensor.data)
		fmt.sbprintf(&b, "],\"data_offsets\":[%d,%d]}}", offset, next)
		offset = next
	}
	strings.write_byte(&b, '}')
	// Pad header with spaces to 8-byte alignment, per safetensors spec.
	for strings.builder_len(b) % 8 != 0 do strings.write_byte(&b, ' ')
	header := strings.to_string(b)

	out := make([]u8, 8 + len(header) + offset, allocator)
	endian.unchecked_put_u64le(out[:8], u64(len(header)))
	copy(out[8:], header)
	pos := 8 + len(header)
	for e in entries {
		copy(out[pos:], e.tensor.data)
		pos += len(e.tensor.data)
	}
	return out
}

