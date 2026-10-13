package pocket

import "core:strings"
import "core:unicode"
import "core:unicode/utf8"

// A word of the source text and when it is spoken.
Word :: struct {
	text:      string, // as written in the source text
	index:     int, // position among the source text's words
	start:     f64, // seconds
	end:       f64,
	estimated: bool, // not aligned letter by letter: a wildcard ("$3.50"), or the chunk's span shared by length
}

Span :: struct {
	begin, end: int, // bytes
}

@(private) CONTINUATIONS :: "-‐‑'’"

@(private)
is_word_rune :: proc(r: rune) -> bool {
	return unicode.is_letter(r) || unicode.is_number(r)
}

@(private)
is_mark :: proc(r: rune) -> bool {
	return unicode.is_nonspacing_mark(r) || unicode.is_spacing_mark(r) || unicode.is_enclosing_mark(r)
}

// The words of a text: runs of letters and digits (with their combining
// marks), joined across a single hyphen or apostrophe between word
// characters ("don't", "well-known"); "$3.50" is the words "3" and "50".
word_spans :: proc(text: string, allocator := context.temp_allocator) -> [dynamic]Span {
	spans := make([dynamic]Span, allocator)
	i := 0
	for i < len(text) {
		r, n := utf8.decode_rune_in_string(text[i:])
		if !is_word_rune(r) {
			i += n
			continue
		}
		begin := i
		for {
			for i < len(text) {
				r2, n2 := utf8.decode_rune_in_string(text[i:])
				if !is_word_rune(r2) && !is_mark(r2) do break
				i += n2
			}
			if i < len(text) {
				c, cn := utf8.decode_rune_in_string(text[i:])
				if strings.contains_rune(CONTINUATIONS, c) && i + cn < len(text) {
					after, _ := utf8.decode_rune_in_string(text[i + cn:])
					if is_word_rune(after) {
						i += cn
						continue
					}
				}
			}
			break
		}
		append(&spans, Span{begin, i})
	}
	return spans
}
