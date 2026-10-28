// The protobuf wire format, as much of it as ottis reads and writes: the
// SentencePiece model and the ONNX graph's output list. Fields are read in
// order from a byte slice; nothing is decoded until asked for.
package proto

Wire :: enum u8 {
	Varint = 0,
	I64    = 1,
	Len    = 2,
	I32    = 5,
}

Field :: struct {
	number: u32,
	wire:   Wire,
	value:  u64, // Varint, I64 and I32 payloads
	bytes:  []u8, // Len payload
	start:  int, // offset of the tag in the message
	end:    int, // offset just past the payload
}

Reader :: struct {
	data:      []u8,
	pos:       int,
	malformed: bool, // next stopped on bad input, wherever pos is
}

read_varint :: proc(r: ^Reader) -> (value: u64, ok: bool) {
	shift: uint
	for r.pos < len(r.data) && shift < 64 {
		b := r.data[r.pos]
		r.pos += 1
		value |= u64(b & 0x7f) << shift
		if b < 0x80 do return value, true
		shift += 7
	}
	return 0, false
}

// The next field, or ok = false at the end of the message or on malformed
// input (`finished` tells the two apart).
next :: proc(r: ^Reader) -> (f: Field, ok: bool) {
	if r.pos >= len(r.data) do return {}, false
	defer if !ok do r.malformed = true
	f.start = r.pos
	tag := read_varint(r) or_return
	f.number = u32(tag >> 3)
	f.wire = Wire(tag & 7)
	switch f.wire {
	case .Varint:
		f.value = read_varint(r) or_return
	case .I64:
		if len(r.data) - r.pos < 8 do return {}, false
		for i in 0 ..< 8 do f.value |= u64(r.data[r.pos + i]) << (8 * uint(i))
		r.pos += 8
	case .I32:
		if len(r.data) - r.pos < 4 do return {}, false
		for i in 0 ..< 4 do f.value |= u64(r.data[r.pos + i]) << (8 * uint(i))
		r.pos += 4
	case .Len:
		n := read_varint(r) or_return
		if n > u64(len(r.data) - r.pos) do return {}, false
		f.bytes = r.data[r.pos:][:n]
		r.pos += int(n)
	case:
		return {}, false
	}
	f.end = r.pos
	return f, true
}

// Whether a reader stopped at the end of its message rather than on bad
// input (a value cut off by the end of the message is bad input).
finished :: proc(r: Reader) -> bool {
	return !r.malformed && r.pos == len(r.data)
}

as_f32 :: proc(f: Field) -> f32 {
	return transmute(f32)u32(f.value)
}

append_varint :: proc(buf: ^[dynamic]u8, value: u64) {
	v := value
	for v >= 0x80 {
		append(buf, u8(v) | 0x80)
		v >>= 7
	}
	append(buf, u8(v))
}

append_tag :: proc(buf: ^[dynamic]u8, number: u32, wire: Wire) {
	append_varint(buf, u64(number) << 3 | u64(wire))
}

append_len :: proc(buf: ^[dynamic]u8, number: u32, payload: []u8) {
	append_tag(buf, number, .Len)
	append_varint(buf, u64(len(payload)))
	append(buf, ..payload)
}

append_uint :: proc(buf: ^[dynamic]u8, number: u32, value: u64) {
	append_tag(buf, number, .Varint)
	append_varint(buf, value)
}
