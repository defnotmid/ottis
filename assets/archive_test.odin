package assets

import "core:hash"
import "core:mem"
import "core:testing"

// A zip with one entry, `name`, whose central record says it is `packed`
// bytes stored (or deflated, `method` 8) as `size` with CRC-32 `crc`, and
// that its name is `name_len` long.
@(private = "file")
one_entry_zip :: proc(name: string, data: []u8, packed, size: u32, name_len: u16, crc: u32, method: u16 = 0) -> []u8 {
	buf := make([dynamic]u8, context.temp_allocator)
	u16le :: proc(b: ^[dynamic]u8, v: u16) {append(b, u8(v), u8(v >> 8))}
	u32le :: proc(b: ^[dynamic]u8, v: u32) {append(b, u8(v), u8(v >> 8), u8(v >> 16), u8(v >> 24))}
	zeros :: proc(b: ^[dynamic]u8, n: int) {for _ in 0 ..< n do append(b, 0)}
	u32le(&buf, 0x04034b50)
	zeros(&buf, 22)
	u16le(&buf, u16(len(name)))
	u16le(&buf, 0)
	append(&buf, name)
	append(&buf, ..data)
	central := len(buf)
	u32le(&buf, 0x02014b50)
	zeros(&buf, 6)
	u16le(&buf, method)
	zeros(&buf, 4)
	u32le(&buf, crc)
	u32le(&buf, packed)
	u32le(&buf, size)
	u16le(&buf, name_len)
	zeros(&buf, 12)
	u32le(&buf, 0) // local header at 0
	append(&buf, name)
	u32le(&buf, 0x06054b50)
	zeros(&buf, 6)
	u16le(&buf, 1)
	zeros(&buf, 4)
	u32le(&buf, u32(central))
	u16le(&buf, 0)
	return buf[:]
}

// A central record whose name runs past the archive, and a stored member
// whose sizes disagree, are refused with nothing left allocated; a good
// stored member is extracted.
@(test)
zip_refuses_bad_records :: proc(t: ^testing.T) {
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, context.allocator)
	defer mem.tracking_allocator_destroy(&track)
	context.allocator = mem.tracking_allocator(&track)

	NAME :: "lib/x.so"
	members := []string{"lib/x.so"}
	data := []u8{1, 2, 3}
	crc := hash.crc32(data)
	files, err := extract(one_entry_zip(NAME, data, 3, 3, 60000, crc), members)
	testing.expectf(t, err != nil && files == nil, "name past the end: %v", err)
	delete(err.?)
	files, err = extract(one_entry_zip(NAME, data, 3, 5, len(NAME), crc), members)
	testing.expect_value(t, err.?, "zip: stored lib/x.so is 3 bytes, expected 5")
	testing.expect(t, files == nil)
	delete(err.?)
	testing.expectf(t, len(track.allocation_map) == 0, "%d allocations left", len(track.allocation_map))

	files, err = extract(one_entry_zip(NAME, data, 3, 3, len(NAME), crc + 1), members)
	testing.expect_value(t, err.?, "zip: lib/x.so fails its CRC-32")
	testing.expect(t, files == nil)
	delete(err.?)
	testing.expectf(t, len(track.allocation_map) == 0, "%d allocations left", len(track.allocation_map))

	files, err = extract(one_entry_zip(NAME, data, 3, 3, len(NAME), crc), members)
	testing.expect(t, err == nil && len(files) == 1 && len(files[0]) == 3 && files[0][2] == 3)
	for f in files do delete(f)
	delete(files)
	testing.expect_value(t, len(track.allocation_map), 0)

	// Deflated and empty: not inflated, however much data follows.
	junk := []u8{0x4b, 0x4c, 0x4a, 0x06, 0, 0, 0, 0, 0, 0}
	files, err = extract(one_entry_zip(NAME, junk, u32(len(junk)), 0, len(NAME), 0, 8), members)
	testing.expectf(t, err == nil && len(files) == 1 && len(files[0]) == 0, "empty deflated member: %v", err)
	for f in files do delete(f)
	delete(files)
	testing.expect_value(t, len(track.allocation_map), 0)

	// zip's CRC-32 is core:hash's: the check value of "123456789".
	testing.expect_value(t, hash.crc32(transmute([]u8)string("123456789")), 0xcbf43926)
}
