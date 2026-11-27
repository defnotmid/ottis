package assets

import "core:compress/deflate"
import "core:encoding/endian"
import "core:fmt"
import "core:hash"
import "core:slice"

// The named members of a zip archive, matched by their full path. Every
// member must be found once (empty ones too); a name that appears twice is
// refused. On an error nothing is returned.
extract :: proc(archive: []u8, members: []string) -> ([][]u8, Error) {
	files := make([][]u8, len(members))
	found := make([]bool, len(members), context.temp_allocator)
	err := extract_zip(archive, members, files, found)
	if err == nil {
		for f, i in found do if !f {
			err = fmt.aprintf("archive has no %s", members[i])
			break
		}
	}
	if err != nil {
		for f in files do delete(f)
		delete(files)
		return nil, err
	}
	return files, nil
}

@(private)
extract_zip :: proc(zip: []u8, members: []string, files: [][]u8, found: []bool) -> Error {
	// End of central directory: the last "PK\x05\x06" within the trailing
	// comment's maximum length.
	eocd := -1
	for i := len(zip) - 22; i >= max(0, len(zip) - 22 - 65535); i -= 1 {
		if endian.unchecked_get_u32le(zip[i:]) == 0x06054b50 {
			eocd = i
			break
		}
	}
	if eocd < 0 do return fmt.aprintf("zip: no central directory")
	count := int(endian.unchecked_get_u16le(zip[eocd + 10:]))
	pos := int(endian.unchecked_get_u32le(zip[eocd + 16:]))
	for _ in 0 ..< count {
		if pos + 46 > len(zip) || endian.unchecked_get_u32le(zip[pos:]) != 0x02014b50 do return fmt.aprintf("zip: bad central entry at %d", pos)
		method := endian.unchecked_get_u16le(zip[pos + 10:])
		crc := endian.unchecked_get_u32le(zip[pos + 16:])
		packed := int(endian.unchecked_get_u32le(zip[pos + 20:]))
		size := int(endian.unchecked_get_u32le(zip[pos + 24:]))
		name_len := int(endian.unchecked_get_u16le(zip[pos + 28:]))
		extra_len := int(endian.unchecked_get_u16le(zip[pos + 30:]))
		comment_len := int(endian.unchecked_get_u16le(zip[pos + 32:]))
		local := int(endian.unchecked_get_u32le(zip[pos + 42:]))
		if name_len + extra_len + comment_len > len(zip) - pos - 46 do return fmt.aprintf("zip: bad central entry at %d", pos)
		name := string(zip[pos + 46:][:name_len])
		pos += 46 + name_len + extra_len + comment_len

		i := slice.linear_search(members, name) or_continue
		if found[i] do return fmt.aprintf("zip: %s appears twice", name)
		found[i] = true
		if local > len(zip) - 30 || endian.unchecked_get_u32le(zip[local:]) != 0x04034b50 do return fmt.aprintf("zip: bad local header for %s", name)
		start := local + 30 + int(endian.unchecked_get_u16le(zip[local + 26:])) + int(endian.unchecked_get_u16le(zip[local + 28:]))
		if start + packed > len(zip) do return fmt.aprintf("zip: %s is truncated", name)
		data := zip[start:][:packed]
		switch method {
		case 0:
			if packed != size do return fmt.aprintf("zip: stored %s is %d bytes, expected %d", name, packed, size)
			files[i] = make([]u8, size)
			copy(files[i], data)
		case 8:
			// An empty member is not inflated: with no size to stop at,
			// inflating would run to the end of whatever the data holds.
			if size == 0 do break
			out: [dynamic]u8
			if zerr := deflate.inflate_raw(&out, data, expected_output_size = size); zerr != nil {
				delete(out)
				return fmt.aprintf("zip: inflating %s: %v", name, zerr)
			}
			files[i] = out[:]
			if len(files[i]) != size do return fmt.aprintf("zip: %s inflated to %d bytes, expected %d", name, len(files[i]), size)
		case:
			return fmt.aprintf("zip: %s uses compression method %d", name, method)
		}
		if hash.crc32(files[i]) != crc do return fmt.aprintf("zip: %s fails its CRC-32", name)
	}
	return nil
}
