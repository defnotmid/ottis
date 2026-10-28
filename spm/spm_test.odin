package spm

import "core:log"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:testing"
import "../proto"

// Ids from the reference SentencePiece (0.2.0) on english_2026-04's
// tokenizer.model. The model is fetched data; without it the test skips.
@(test)
encode_matches_sentencepiece :: proc(t: ^testing.T) {
	root := os.get_env("OTTIS_DATA", context.temp_allocator)
	if root == "" do root = ".data"
	data, err := os.read_entire_file(tokenizer_path(root), context.allocator)
	if err != nil {
		log.infof("skipped: no tokenizer under %s (run `ottis fetch`)", root)
		return
	}
	defer delete(data)
	m, load_err := load(data)
	testing.expect_value(t, load_err, "")
	defer destroy(&m)

	Case :: struct {
		text: string,
		ids:  []i32,
	}
	cases := []Case {
		{"Hello world, this is a timing test.", {2994, 578, 262, 285, 277, 267, 427, 711, 273, 1115, 263}},
		{
			"Dr. Smith paid $3.50 on 12/03 — naïve café!",
			{1555, 263, 3669, 2286, 1124, 450, 263, 437, 316, 288, 260, 341, 365, 51, 316, 450, 260, 3133, 913, 199, 179, 314, 331, 2250, 745, 682},
		},
		{"ＡＢＣ\u3000ﬁ x²", {260, 243, 192, 165, 243, 192, 166, 243, 192, 167, 231, 132, 132, 243, 176, 133, 260, 568, 198, 182}},
		// Bytes outside valid UTF-8 (a stray byte, a cut sequence, a
		// surrogate, an overlong) are each one U+FFFD, as SentencePiece's
		// normalizer makes them, and the text after them is untouched.
		{"Bad\xffbyte.", {1076, 307, 243, 195, 193, 1498, 502, 263}},
		{"Cut \xe2\x82 here.", {3018, 274, 260, 243, 195, 193, 243, 195, 193, 375, 263}},
		{"Lone \x80 cont.", {963, 455, 260, 243, 195, 193, 1757, 263}},
		{"Ok \xef\xbf\xbd real.", {705, 518, 260, 243, 195, 193, 584, 263}},
		{"Surrogate \xed\xa0\x80 x.", {446, 513, 489, 453, 492, 260, 243, 195, 193, 243, 195, 193, 243, 195, 193, 260, 568, 263}},
		{"Overlong \xc0\xaf x.", {3398, 589, 808, 260, 243, 195, 193, 243, 195, 193, 260, 568, 263}},
	}
	tokens: [dynamic]Token
	defer delete(tokens)
	for c in cases {
		clear(&tokens)
		encode(&m, c.text, &tokens)
		ids := make([]i32, len(tokens), context.temp_allocator)
		for tok, i in tokens do ids[i] = tok.id
		testing.expectf(t, slice_equal(ids, c.ids), "%q:\n got %v\nwant %v", c.text, ids, c.ids)
		for tok in tokens {
			testing.expect(t, 0 <= tok.begin && tok.begin <= tok.end && tok.end <= len(c.text))
		}
	}

	// Spans: "▁Hello" covers "Hello", "▁world" covers " world".
	clear(&tokens)
	encode(&m, "Hello world", &tokens)
	testing.expect_value(t, tokens[0].begin, 0)
	testing.expect_value(t, tokens[0].end, 5)
	testing.expect_value(t, tokens[1].begin, 5)
	testing.expect_value(t, tokens[1].end, 11)

	ids := []i32{2994, 578, 262}
	testing.expect_value(t, decode(&m, ids, context.temp_allocator), "Hello world,")
	testing.expect_value(t, decode(&m, ids[1:], context.temp_allocator), "world,")
}

@(private)
tokenizer_path :: proc(root: string) -> string {
	path, _ := filepath.join({root, "english_2026-04", "tokenizer.model"}, context.temp_allocator)
	return path
}

@(private)
slice_equal :: proc(a, b: []i32) -> bool {
	if len(a) != len(b) do return false
	for v, i in a do if v != b[i] do return false
	return true
}

// A model file outside the shape load reads is refused, and what load
// allocated before it found out is freed: a piece cut short, a piece type
// SentencePiece does not have, an unk_id past the pieces. The same model
// with a good unk_id loads.
@(test)
load_refuses_malformed_models :: proc(t: ^testing.T) {
	model :: proc(unk_id: u64, piece_type: u64 = 1) -> []u8 {
		buf := make([dynamic]u8, context.temp_allocator)
		for p, i in ([]string{"<unk>", "a"}) {
			piece := make([dynamic]u8, context.temp_allocator)
			proto.append_len(&piece, 1, transmute([]u8)p)
			proto.append_tag(&piece, 2, .I32)
			append(&piece, 0, 0, 0x80, 0xbf) // -1.0
			proto.append_uint(&piece, 3, 2 if i == 0 else piece_type)
			proto.append_len(&buf, 1, piece[:])
		}
		trainer := make([dynamic]u8, context.temp_allocator)
		proto.append_uint(&trainer, 3, 1)
		proto.append_uint(&trainer, 40, unk_id)
		proto.append_len(&buf, 2, trainer[:])
		normalizer := make([dynamic]u8, context.temp_allocator)
		proto.append_len(&normalizer, 1, transmute([]u8)string("identity"))
		proto.append_uint(&normalizer, 4, 0)
		proto.append_uint(&normalizer, 5, 1)
		proto.append_len(&buf, 3, normalizer[:])
		return buf[:]
	}
	cut := make([dynamic]u8, context.temp_allocator)
	append(&cut, ..model(0))
	proto.append_len(&cut, 1, {0x08}) // a piece whose type has no value

	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, context.allocator)
	defer mem.tracking_allocator_destroy(&track)
	Case :: struct {
		data: []u8,
		err:  string,
	}
	for c in ([]Case{{cut[:], "tokenizer: malformed piece"}, {model(0, 9), "tokenizer: unknown piece type"}, {model(2), "tokenizer: unk_id out of range"}}) {
		m, err := load(c.data, mem.tracking_allocator(&track))
		testing.expect_value(t, err, c.err)
		testing.expect_value(t, len(m.pieces), 0)
		testing.expectf(t, len(track.allocation_map) == 0, "%s: %d allocations left", c.err, len(track.allocation_map))
	}
	context.allocator = mem.tracking_allocator(&track)
	m, err := load(model(0))
	testing.expect_value(t, err, "")
	testing.expect_value(t, len(m.pieces), 2)
	destroy(&m)
	testing.expect_value(t, len(track.allocation_map), 0)
}
