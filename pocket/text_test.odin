package pocket

import "core:log"
import "core:os"
import "core:testing"
import "../assets"
import "../spm"

// Sentences split where upstream splits them (one chunk each at one token
// a chunk): after . ! ? and ..., not at a period between digits, nor (as
// upstream) at "3. 5" whose next text starts with a digit. Needs the
// bundle's tokenizer (`ottis fetch`); skipped without it.
@(test)
chunks_split_on_sentences_not_decimals :: proc(t: ^testing.T) {
	root, _ := assets.data_dir(context.temp_allocator)
	data, err := os.read_entire_file(assets.path_in(root, assets.BUNDLE_NAME + "/tokenizer.model"), context.allocator)
	if err != nil {
		log.infof("skipped: no tokenizer in %s (ottis fetch)", root)
		return
	}
	defer delete(data)
	tok, terr := spm.load(data)
	if !testing.expect_value(t, terr, "") do return
	defer spm.destroy(&tok)

	chunks, ok := split_chunks(&tok, "It costs 3.50 today! Pi is 3.14... Version 3. 5 apples? Yes.", 1, true)
	testing.expect(t, ok)
	want := []string{"It costs 3.50 today!", "Pi is 3.14...", "Version 3. 5 apples?", "Yes."}
	if !testing.expectf(t, len(chunks) == len(want), "%d chunks: %q", len(chunks), chunks[:]) do return
	for c, i in chunks do testing.expect_value(t, c, want[i])
}
