package spm

import "core:log"
import "core:os"
import "core:path/filepath"
import "core:testing"

// Ids from the reference SentencePiece (0.2.x) on english_2026-04's
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
