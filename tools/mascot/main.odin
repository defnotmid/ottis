// Ottis, the mascot: a 32x32 pixel-art otter holding a microphone, in the
// family of onimite's Pip, odecs' turntable and dangersnail's snail. Writes
// an SVG whose parts are groups a script shows or hides by id (onimite:
// svg{ overrides = { mouth_open = { visible = true }, mouth_closed = { visible = false } } }):
//
//   body          outline, fur, shade, highlight, muzzle, belly, cheeks, ears,
//                 paws and the microphone (always on)
//   eyes_open     big eyes with two shines (on)    eyes_closed  a happy blink
//   eyes_o        round surprised eyes
//   mouth_closed  a small smile (on)               mouth_open   "ah": open, tongue
//   mouth_o       "oo": a small round mouth        mouth_ee     "ee": wide, teeth
//   waves_a       one sound wave                   waves_b      two waves
//                 (both off: show them while Ottis talks)
//
// The mouths are speech shapes, so ottis's word cues can make Ottis talk.
// Also writes the project logo as PNGs, scaled without smoothing:
//
//   odin run tools/mascot -- art   (art/logo.svg, logo.png 256 px, logo_32x32.png)
package mascot

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strings"
import stbi "vendor:stb/image"

W :: 32
H :: 32

Grid :: [H][W]u8 // palette keys, '.' empty

color :: proc(c: u8) -> string {
	switch c {
	case 'O': return "#2a1a14" // outline
	case 'B': return "#9a6236" // fur
	case 'D': return "#6f4325" // fur shade
	case 'H': return "#c48a55" // fur highlight
	case 'C': return "#f4dcb4" // muzzle and belly
	case 'c': return "#dcbb8e" // muzzle shade
	case 'E': return "#1b1220" // eyes, nose
	case 'W': return "#ffffff" // shines
	case 'K': return "#ff9db4" // cheeks
	case 'M': return "#5a1f24" // mouth
	case 'T': return "#ff7a8a" // tongue
	case 'G': return "#d4d6e4" // microphone grille
	case 'g': return "#8c8fa8" // grille shade
	case 'R': return "#2b2d42" // microphone handle
	case 'r': return "#4a4e6e" // handle highlight
	case 'S': return "#5eead4" // sound waves
	case 's': return "#22b8cf"
	}
	return "#ff00ff"
}

empty :: proc() -> (g: Grid) {
	for &row in g do for &p in row do p = '.'
	return
}

put :: proc(g: ^Grid, x, y: int, c: u8) {
	if 0 <= x && x < W && 0 <= y && y < H do g[y][x] = c
}

in_ellipse :: proc(x, y: int, cx, cy, rx, ry: f64) -> (bool, f64, f64) {
	u := (f64(x) + 0.5 - cx) / rx
	v := (f64(y) + 0.5 - cy) / ry
	return u * u + v * v <= 1, u, v
}

// Fills an ellipse: shade along the lower right rim, highlight upper left.
fur :: proc(g: ^Grid, cx, cy, rx, ry: f64) {
	for y in 0 ..< H do for x in 0 ..< W {
		inside, u, v := in_ellipse(x, y, cx, cy, rx, ry)
		if !inside do continue
		c := u8('B')
		if u * u + v * v > 0.6 && u * 0.5 + v * 0.85 > 0.35 do c = 'D'
		put(g, x, y, c)
	}
}

// Outlines every filled pixel's empty 4-neighbours.
outline :: proc(g: ^Grid) {
	src := g^
	for y in 0 ..< H do for x in 0 ..< W {
		if src[y][x] != '.' do continue
		for d in ([4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
			nx, ny := x + d.x, y + d.y
			if 0 <= nx && nx < W && 0 <= ny && ny < H && src[ny][nx] != '.' && src[ny][nx] != 'O' {
				g[y][x] = 'O'
				break
			}
		}
	}
}

// Draws `fg` over `g`, outlining fg's shapes against everything else.
over :: proc(g: ^Grid, fg: Grid) {
	shape := fg
	outline(&shape)
	for y in 0 ..< H do for x in 0 ..< W do if shape[y][x] != '.' do g[y][x] = shape[y][x]
}

draw_body :: proc() -> Grid {
	g := empty()
	// Shoulders, cut by the bottom edge.
	shoulders := empty()
	fur(&shoulders, 16, 28.5, 11.5, 7.5)
	// a pale chest under the microphone
	for y in 0 ..< H do for x in 0 ..< W {
		if inside, _, _ := in_ellipse(x, y, 16, 31, 6.5, 7); inside && shoulders[y][x] != '.' do put(&shoulders, x, y, 'c')
	}
	over(&g, shoulders)

	// The head over them: ear nubs low on the sides, a broad skull, a sea
	// otter's pale face (shaded at the jaw) under a brown crown.
	head := empty()
	fur(&head, 6.3, 8.6, 1.9, 1.9)
	fur(&head, 25.7, 8.6, 1.9, 1.9)
	fur(&head, 16, 12.3, 10.6, 8.0)
	put(&head, 6, 8, 'D')
	put(&head, 25, 8, 'D')
	for y in 0 ..< H do for x in 0 ..< W {
		inside, _, v := in_ellipse(x, y, 16, 13.8, 8.8, 6.4)
		if inside do put(&head, x, y, 'c' if v > 0.62 else 'C')
	}
	for p in ([?][2]int{{12, 5}, {13, 5}, {14, 5}, {10, 6}, {11, 6}, {8, 7}, {9, 7}, {7, 9}}) do put(&head, p.x, p.y, 'H')
	for p in ([?][2]int{{7, 14}, {8, 14}, {23, 14}, {24, 14}}) do put(&head, p.x, p.y, 'K')
	// The big nose with a shine; whisker dots out on the cheeks.
	for p in ([?][2]int{{14, 13}, {15, 13}, {16, 13}, {17, 13}, {15, 14}, {16, 14}}) do put(&head, p.x, p.y, 'E')
	put(&head, 15, 13, 'W')
	for p in ([?][2]int{{9, 16}, {10, 17}, {22, 16}, {21, 17}}) do put(&head, p.x, p.y, 'c')
	over(&g, head)

	// The microphone held up in front of the chest, a little below the
	// mouth: a round silver head with a shine and a few grille holes, a teal
	// band at its neck, the black handle in both paws.
	fg := empty()
	for y in 20 ..< 25 do for x in 13 ..< 19 {
		if (y == 20 || y == 24) && (x == 13 || x == 18) do continue
		c := u8('G')
		if x == 18 || y == 24 || (x == 17 && y == 23) do c = 'g'
		else if (x + y) % 2 == 0 && x > 13 && y > 20 && !(x == 14 && y == 21) do c = 'g'
		put(&fg, x, y, c)
	}
	put(&fg, 14, 21, 'W')
	put(&fg, 15, 21, 'W')
	for x in 14 ..< 18 do put(&fg, x, 25, 'S' if x < 17 else 's')
	for y in 26 ..< 30 do for x in 15 ..< 17 do put(&fg, x, y, 'r' if x == 15 else 'R')
	for p in ([?][2]int{{12, 26}, {13, 26}, {14, 26}, {12, 27}, {13, 27}, {14, 27}, {13, 28}, {14, 28}}) do put(&fg, p.x, p.y, 'B')
	for p in ([?][2]int{{17, 26}, {18, 26}, {19, 26}, {17, 27}, {18, 27}, {19, 27}, {17, 28}, {18, 28}}) do put(&fg, p.x, p.y, 'B')
	for p in ([?][2]int{{13, 26}, {18, 26}}) do put(&fg, p.x, p.y, 'H')
	for p in ([?][2]int{{14, 27}, {17, 27}}) do put(&fg, p.x, p.y, 'D')
	over(&g, fg)
	return g
}

draw_eyes :: proc() -> (open, closed, round: Grid) {
	open, closed, round = empty(), empty(), empty()
	for ex in ([2]int{10, 19}) {
		for y in 9 ..< 12 do for x in ex ..< ex + 3 do put(&open, x, y, 'E')
		put(&open, ex, 9, 'W')
		put(&open, ex + 2, 11, 'W')
		// a happy closed eye: a small arch
		put(&closed, ex, 11, 'E')
		put(&closed, ex + 1, 10, 'E')
		put(&closed, ex + 2, 11, 'E')
		// surprised: wider, one big shine
		for y in 8 ..< 13 do for x in ex - 1 ..< ex + 4 {
			if (y == 8 || y == 12) && (x == ex - 1 || x == ex + 3) do continue
			put(&round, x, y, 'E')
		}
		put(&round, ex, 9, 'W')
		put(&round, ex + 1, 9, 'W')
		put(&round, ex, 10, 'W')
	}
	return
}

draw_mouths :: proc() -> (closed, open, o, ee: Grid) {
	closed, open, o, ee = empty(), empty(), empty(), empty()
	// a "w" under the nose
	for p in ([?][2]int{{13, 15}, {14, 16}, {15, 15}, {16, 15}, {17, 16}, {18, 15}}) do put(&closed, p.x, p.y, 'M')
	// "ah": open, the tongue showing
	for p in ([?][2]int{{14, 15}, {15, 15}, {16, 15}, {17, 15}, {14, 16}, {17, 16}, {15, 17}, {16, 17}}) do put(&open, p.x, p.y, 'M')
	put(&open, 15, 16, 'T')
	put(&open, 16, 16, 'T')
	// "oo": small and round
	for p in ([?][2]int{{15, 15}, {16, 15}, {15, 16}, {16, 16}}) do put(&o, p.x, p.y, 'M')
	// "ee": wide, teeth
	for x in 13 ..< 19 do put(&ee, x, 15, 'M')
	for x in 14 ..< 18 do put(&ee, x, 16, 'W')
	put(&ee, 13, 16, 'M')
	put(&ee, 18, 16, 'M')
	return
}

draw_waves :: proc() -> (one, two: Grid) {
	one, two = empty(), empty()
	arc :: proc(g: ^Grid, x0, y0, h: int, c: u8) {
		// ")" : a column bowed to the right
		for i in 0 ..< h {
			off := 1 if i > 0 && i < h - 1 else 0
			put(g, x0 + off, y0 + i, c)
		}
	}
	arc(&one, 26, 1, 5, 'S')
	arc(&two, 26, 1, 5, 'S')
	arc(&two, 28, 0, 7, 's')
	return
}

Layer :: struct {
	name: string,
	grid: Grid,
	on:   bool,
}

svg :: proc(layers: []Layer) -> string {
	b := strings.builder_make()
	fmt.sbprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" shape-rendering="crispEdges">`, W, H, W * 8, H * 8)
	strings.write_byte(&b, '\n')
	for l in layers {
		fmt.sbprintf(&b, `<g id="%s"%s>`, l.name, "" if l.on else ` display="none"`)
		for y in 0 ..< H do for x in 0 ..< W {
			c := l.grid[y][x]
			if c == '.' do continue
			fmt.sbprintf(&b, `<rect x="%d" y="%d" width="1" height="1" fill="%s"/>`, x, y, color(c))
		}
		strings.write_string(&b, "</g>\n")
	}
	strings.write_string(&b, "</svg>\n")
	return strings.to_string(b)
}

hex_byte :: proc(s: string) -> u8 {
	v: u8
	for c in transmute([]u8)s {
		v <<= 4
		switch c {
		case '0' ..= '9': v |= c - '0'
		case 'a' ..= 'f': v |= c - 'a' + 10
		}
	}
	return v
}

// The visible layers flattened, scaled by `scale` without smoothing.
png :: proc(path: string, layers: []Layer, scale: int) -> bool {
	size := W * scale
	pixels := make([]u8, size * size * 4)
	defer delete(pixels)
	for l in layers {
		if !l.on do continue
		for y in 0 ..< H do for x in 0 ..< W {
			c := l.grid[y][x]
			if c == '.' do continue
			hex := color(c)
			rgba := [4]u8{hex_byte(hex[1:3]), hex_byte(hex[3:5]), hex_byte(hex[5:7]), 255}
			for dy in 0 ..< scale do for dx in 0 ..< scale {
				i := ((y * scale + dy) * size + x * scale + dx) * 4
				pixels[i + 0], pixels[i + 1], pixels[i + 2], pixels[i + 3] = rgba[0], rgba[1], rgba[2], rgba[3]
			}
		}
	}
	return stbi.write_png(strings.clone_to_cstring(path, context.temp_allocator), i32(size), i32(size), 4, raw_data(pixels), i32(size * 4)) != 0
}

main :: proc() {
	dir := os.args[1] if len(os.args) > 1 else "art"
	eyes_open, eyes_closed, eyes_o := draw_eyes()
	mouth_closed, mouth_open, mouth_o, mouth_ee := draw_mouths()
	waves_a, waves_b := draw_waves()
	layers := []Layer {
		{"body", draw_body(), true},
		{"eyes_open", eyes_open, true},
		{"eyes_closed", eyes_closed, false},
		{"eyes_o", eyes_o, false},
		{"mouth_closed", mouth_closed, true},
		{"mouth_open", mouth_open, false},
		{"mouth_o", mouth_o, false},
		{"mouth_ee", mouth_ee, false},
		{"waves_a", waves_a, false},
		{"waves_b", waves_b, false},
	}
	join :: proc(dir, name: string) -> string {
		p, _ := filepath.join({dir, name}, context.temp_allocator)
		return p
	}
	if err := os.write_entire_file(join(dir, "logo.svg"), svg(layers)); err != nil {
		fmt.eprintfln("writing %s: %v", join(dir, "logo.svg"), err)
		os.exit(1)
	}
	if !png(join(dir, "logo.png"), layers, 8) || !png(join(dir, "logo_32x32.png"), layers, 1) {
		fmt.eprintln("writing the PNGs failed")
		os.exit(1)
	}
}
