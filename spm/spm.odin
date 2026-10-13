// SentencePiece Unigram encoding, for the models Pocket TTS ships: identity
// normalizer, a dummy-prefix "▁", spaces escaped as "▁", byte fallback for
// characters no piece covers. The model file is read straight from its
// protobuf; anything outside that shape is refused at load instead of
// encoded differently.
package spm

import "core:strings"
import "core:unicode/utf8"
import "../proto"

SPACE :: "▁" // "▁"

Piece_Type :: enum u8 {
	Normal       = 1,
	Unknown      = 2,
	Control      = 3,
	User_Defined = 4,
	Unused       = 5,
	Byte         = 6,
}

Model :: struct {
	pieces:           []string, // by id
	scores:           []f32,
	types:            []Piece_Type,
	ids:              map[string]i32, // pieces that can match text
	byte_ids:         [256]i32, // <0xNN> pieces
	unk_id:           i32,
	min_score:        f32,
	max_score:        f32,
	max_piece_bytes:  int,
	add_dummy_prefix: bool,
	byte_fallback:    bool,
	backing:          []u8,
}

// One token and the bytes of the source text it covers. A token for part
// of a character (byte fallback) covers the whole character; the dummy
// prefix covers nothing.
Token :: struct {
	id:    i32,
	begin: int,
	end:   int,
}

UNK_PENALTY :: 10.0

load :: proc(data: []u8, allocator := context.allocator) -> (m: Model, err: string) {
	context.allocator = allocator
	m.backing = make([]u8, len(data))
	copy(m.backing, data)

	pieces: [dynamic]string
	scores: [dynamic]f32
	types: [dynamic]Piece_Type
	model_type: u64 = 1
	normalizer := "nmt_nfkc"
	charsmap_bytes := 0
	m.add_dummy_prefix = true
	escape_whitespaces := true
	remove_extra_whitespaces := true
	m.unk_id = 0

	r := proto.Reader{data = m.backing}
	for f in proto.next(&r) {
		switch f.number {
		case 1: // SentencePiece
			piece: string
			score: f32
			type := Piece_Type.Normal
			pr := proto.Reader{data = f.bytes}
			for pf in proto.next(&pr) {
				switch pf.number {
				case 1: piece = string(pf.bytes)
				case 2: score = proto.as_f32(pf)
				case 3: type = Piece_Type(pf.value)
				}
			}
			append(&pieces, piece)
			append(&scores, score)
			append(&types, type)
		case 2: // TrainerSpec
			tr := proto.Reader{data = f.bytes}
			for tf in proto.next(&tr) {
				switch tf.number {
				case 3: model_type = tf.value
				case 35: m.byte_fallback = tf.value != 0
				case 40: m.unk_id = i32(tf.value)
				}
			}
		case 3: // NormalizerSpec
			nr := proto.Reader{data = f.bytes}
			for nf in proto.next(&nr) {
				switch nf.number {
				case 1: normalizer = string(nf.bytes)
				case 2: charsmap_bytes = len(nf.bytes)
				case 3: m.add_dummy_prefix = nf.value != 0
				case 4: remove_extra_whitespaces = nf.value != 0
				case 5: escape_whitespaces = nf.value != 0
				}
			}
		}
	}
	if !proto.finished(r) do return m, "tokenizer: malformed protobuf"
	if model_type != 1 do return m, "tokenizer: not a Unigram model"
	if normalizer != "identity" || charsmap_bytes != 0 do return m, "tokenizer: only the identity normalizer is supported"
	if remove_extra_whitespaces || !escape_whitespaces do return m, "tokenizer: unsupported whitespace handling"
	if len(pieces) == 0 do return m, "tokenizer: no pieces"

	m.pieces = pieces[:]
	m.scores = scores[:]
	m.types = types[:]
	m.min_score = max(f32)
	m.max_score = min(f32)
	for &id in m.byte_ids do id = -1
	for piece, i in m.pieces {
		switch m.types[i] {
		case .Normal:
			m.min_score = min(m.min_score, m.scores[i])
			m.max_score = max(m.max_score, m.scores[i])
			fallthrough
		case .User_Defined:
			m.ids[piece] = i32(i)
			m.max_piece_bytes = max(m.max_piece_bytes, len(piece))
		case .Byte:
			if len(piece) == 6 && strings.has_prefix(piece, "<0x") && piece[5] == '>' {
				v := hex(piece[3]) << 4 | hex(piece[4])
				if v >= 0 do m.byte_ids[v] = i32(i)
			}
		case .Unknown, .Control, .Unused:
		}
	}
	if m.byte_fallback {
		for id in m.byte_ids do if id < 0 do return m, "tokenizer: byte fallback without all 256 byte pieces"
	}
	return m, ""
}

destroy :: proc(m: ^Model) {
	delete(m.pieces)
	delete(m.scores)
	delete(m.types)
	delete(m.ids)
	delete(m.backing)
}

@(private)
hex :: proc(c: u8) -> int {
	switch c {
	case '0' ..= '9': return int(c - '0')
	case 'a' ..= 'f': return int(c - 'a' + 10)
	case 'A' ..= 'F': return int(c - 'A' + 10)
	}
	return -256
}

@(private)
Best :: struct {
	score:     f32,
	starts_at: i32,
	id:        i32,
}

// Viterbi over the normalized text, as SentencePiece's EncodeOptimized:
// positions advance a character at a time, each piece that matches from
// there competes for the position it ends at (first found wins ties,
// shorter pieces are found first), and a character no piece covers
// alone becomes <unk> at min_score - 10.
encode :: proc(m: ^Model, text: string, out: ^[dynamic]Token) {
	// Normalized bytes and the source bytes each came from: the dummy
	// prefix from none, an escaped space from its space.
	norm := make([dynamic]u8, 0, len(text) * 3 + 3, context.temp_allocator)
	src := make([dynamic][2]int, 0, len(text) * 3 + 3, context.temp_allocator)
	if m.add_dummy_prefix && len(text) > 0 {
		append(&norm, SPACE)
		append(&src, [2]int{0, 0}, [2]int{0, 0}, [2]int{0, 0})
	}
	for i := 0; i < len(text); i += 1 {
		if text[i] == ' ' {
			append(&norm, SPACE)
			append(&src, [2]int{i, i + 1}, [2]int{i, i + 1}, [2]int{i, i + 1})
		} else {
			append(&norm, text[i])
			append(&src, [2]int{i, i + 1})
		}
	}
	size := len(norm)

	best := make([]Best, size + 1, context.temp_allocator)
	for &b in best do b.starts_at = -1
	best[0].starts_at = 0
	unk_score := m.min_score - UNK_PENALTY
	for start := 0; start < size; {
		char_len := min(utf8_len(norm[start]), size - start)
		here := best[start].score
		has_single := false
		for end := start + 1; end <= min(size, start + m.max_piece_bytes); end += 1 {
			id, found := m.ids[string(norm[start:end])]
			if !found do continue
			length := end - start
			score := m.scores[id]
			if m.types[id] == .User_Defined do score = f32(length) * m.max_score - 0.1
			candidate := score + here
			if best[end].starts_at == -1 || candidate > best[end].score {
				best[end] = {candidate, i32(start), id}
			}
			if length == char_len do has_single = true
		}
		if !has_single {
			end := start + char_len
			candidate := unk_score + here
			if best[end].starts_at == -1 || candidate > best[end].score {
				best[end] = {candidate, i32(start), m.unk_id}
			}
		}
		start += char_len
	}

	// Backtrack, then emit in order with source spans.
	path := make([dynamic][2]int, 0, size, context.temp_allocator)
	for end := size; end > 0; {
		start := int(best[end].starts_at)
		append(&path, [2]int{start, end})
		end = start
	}
	for i := len(path) - 1; i >= 0; i -= 1 {
		start, end := path[i][0], path[i][1]
		id := best[end].id
		begin_src, end_src := src[start][0], src[end - 1][1]
		if id == m.unk_id && m.byte_fallback {
			for b in norm[start:end] do append(out, Token{m.byte_ids[b], begin_src, end_src})
		} else {
			append(out, Token{id, begin_src, end_src})
		}
	}
}

// Text for a run of ids, as SentencePiece decodes: pieces joined, byte
// pieces as their bytes, "▁" as a space, and the first piece's leading
// space dropped when the model adds a dummy prefix.
decode :: proc(m: ^Model, ids: []i32, allocator := context.allocator) -> string {
	b := strings.builder_make(allocator)
	for id, i in ids {
		if id < 0 || int(id) >= len(m.pieces) do continue
		switch m.types[id] {
		case .Byte:
			for v, byte in m.byte_ids do if v == id do strings.write_byte(&b, u8(byte))
		case .Control, .Unused:
		case .Unknown:
			strings.write_string(&b, " ⁇ ")
		case .Normal, .User_Defined:
			piece := m.pieces[id]
			if i == 0 && m.add_dummy_prefix && strings.has_prefix(piece, SPACE) do piece = piece[len(SPACE):]
			for len(piece) > 0 {
				if strings.has_prefix(piece, SPACE) {
					strings.write_byte(&b, ' ')
					piece = piece[len(SPACE):]
				} else {
					r, n := utf8.decode_rune_in_string(piece)
					strings.write_string(&b, piece[:n])
					_ = r
					piece = piece[n:]
				}
			}
		}
	}
	return strings.to_string(b)
}

@(private)
utf8_len :: proc(first: u8) -> int {
	switch {
	case first < 0x80: return 1
	case first >= 0xf0: return 4
	case first >= 0xe0: return 3
	case first >= 0xc0: return 2
	}
	return 1
}
