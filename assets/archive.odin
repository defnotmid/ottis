package assets

import "core:bytes"
import "core:compress/gzip"
import "core:compress/zlib"
import "core:encoding/endian"
import "core:fmt"
import "core:strings"

// The named members of a release archive (.tgz, or .zip when `is_zip`),
// matched by their path below the archive's top directory. Every member
// must be found as a regular file. On an error nothing is returned.
extract :: proc(archive: []u8, members: []string, is_zip: bool) -> ([][]u8, Error) {
	files := make([][]u8, len(members))
	err := extract_zip(archive, members, files) if is_zip else extract_tgz(archive, members, files)
	if err == nil {
		for f, i in files do if f == nil {
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
extract_tgz :: proc(archive: []u8, members: []string, files: [][]u8) -> Error {
	buf: bytes.Buffer
	defer bytes.buffer_destroy(&buf)
	if gerr := gzip.load_from_bytes(archive, &buf); gerr != nil do return fmt.aprintf("gzip: %v", gerr)
	return extract_tar(bytes.buffer_to_bytes(&buf), members, files)
}

// "./onnxruntime-x/lib/a.so" -> "lib/a.so"
@(private)
below_top :: proc(name: string) -> string {
	n := strings.trim_prefix(name, "./")
	slash := strings.index_byte(n, '/')
	return n[slash + 1:] if slash >= 0 else ""
}

@(private)
match :: proc(name: string, members: []string) -> int {
	rel := below_top(name)
	for m, i in members do if rel == m do return i
	return -1
}

@(private)
octal :: proc(field: []u8) -> int {
	v := 0
	for b in field {
		if b == 0 || b == ' ' {
			if v > 0 do break
			continue
		}
		if b < '0' || b > '7' do return -1
		v = v * 8 + int(b - '0')
	}
	return v
}

@(private)
cstr :: proc(field: []u8) -> string {
	n := bytes.index_byte(field, 0)
	return string(field[:n] if n >= 0 else field)
}

// POSIX ustar with GNU long names ('L') and pax path records ('x').
@(private)
extract_tar :: proc(tar: []u8, members: []string, files: [][]u8) -> Error {
	long_name: string
	for pos := 0; pos + 512 <= len(tar); {
		h := tar[pos:][:512]
		if h[0] == 0 do break // end-of-archive blocks
		size := octal(h[124:136])
		if size < 0 do return fmt.aprintf("tar: bad size at %d", pos)
		data_start := pos + 512
		if data_start + size > len(tar) do return fmt.aprintf("tar: truncated at %d", pos)
		data := tar[data_start:][:size]
		pos = data_start + (size + 511) / 512 * 512

		type := h[156]
		name := cstr(h[0:100])
		if prefix := cstr(h[345:500]); string(h[257:262]) == "ustar" && prefix != "" {
			name = strings.concatenate({prefix, "/", name}, context.temp_allocator)
		}
		switch type {
		case 'L':
			long_name = cstr(data)
			continue
		case 'x':
			records := string(data)
			for record in strings.split_lines_iterator(&records) {
				if i := strings.index(record, " path="); i >= 0 do long_name = record[i + len(" path="):]
			}
			continue
		}
		if long_name != "" {
			name = long_name
			long_name = ""
		}
		if type != '0' && type != 0 do continue
		if i := match(name, members); i >= 0 && files[i] == nil {
			files[i] = make([]u8, size)
			copy(files[i], data)
		}
	}
	return nil
}

@(private)
extract_zip :: proc(zip: []u8, members: []string, files: [][]u8) -> Error {
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
		packed := int(endian.unchecked_get_u32le(zip[pos + 20:]))
		size := int(endian.unchecked_get_u32le(zip[pos + 24:]))
		name_len := int(endian.unchecked_get_u16le(zip[pos + 28:]))
		extra_len := int(endian.unchecked_get_u16le(zip[pos + 30:]))
		comment_len := int(endian.unchecked_get_u16le(zip[pos + 32:]))
		local := int(endian.unchecked_get_u32le(zip[pos + 42:]))
		if name_len + extra_len + comment_len > len(zip) - pos - 46 do return fmt.aprintf("zip: bad central entry at %d", pos)
		name := string(zip[pos + 46:][:name_len])
		pos += 46 + name_len + extra_len + comment_len

		i := match(name, members)
		if i < 0 || files[i] != nil do continue
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
			buf: bytes.Buffer
			if zerr := zlib.inflate_from_byte_array_raw(data, &buf, expected_output_size = size); zerr != nil {
				bytes.buffer_destroy(&buf)
				return fmt.aprintf("zip: inflating %s: %v", name, zerr)
			}
			files[i] = bytes.buffer_to_bytes(&buf)
			if len(files[i]) != size do return fmt.aprintf("zip: %s inflated to %d bytes, expected %d", name, len(files[i]), size)
		case:
			return fmt.aprintf("zip: %s uses compression method %d", name, method)
		}
	}
	return nil
}
