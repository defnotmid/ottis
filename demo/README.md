# ottis demo

A 28 second showcase of ottis, voices and music included: 1920x1080, 60 fps. Ottis, the
mascot, says what ottis does in four of its voices; its mouth, the captions and the word
table all read the word times ottis returns. It is an [onimite](https://github.com/defnotmid/onimite)
script (Odin + Lua video framework, whose `say()` speaks through ottis), not part of ottis's
build.

With onimite's release binary (`onimite update` keeps it current), from ottis's root:

```sh
onimite render demo/video.lua -o demo/demo.mp4 --qp 22   # render
onimite preview demo/video.lua                           # play, hot reload on save
```

The first run downloads ottis's model and the four voices, and odawg's sample sets for the
score; after that it renders offline.

| line | voice | what |
| --- | --- | --- |
| 1-2 | alba | Ottis says hello, then that it tells you when every word is said: the word table lights each word as it is said, with its start and stop |
| 3 | marius | each voice is a few megabytes, fetched on first use |
| 4 | eve | it all runs on the CPU, faster than real time |
| 5 | george | no Python, nothing to install, one small binary |
| 6 | alba | captions, lip sync and cues follow the words |
| end | | the end card, the score resolving under it |

The lines are laid end to end from their own lengths (`line.duration`: 0.8 s before the
first, 0.5 s between, 3.2 s of end card), so a voice that speaks faster or slower moves
everything after it, the score's end included.

## Files

- `video.lua`: the lines (`say()`), Ottis's mouth from the word times (a shape per vowel
  group), the word table, the captions, the end card; the score on a `music` bus `ducked`
  under every line.
- `track.lua`: the score, an odawg `song{}`. Jazz-hop after Nujabes at 86 BPM in D major,
  a bed under the voice rather than a tune: a swung boom-bap on the virtuosity kit, an
  upright bass (Karoryfer's growlybass), an electric piano of its own (an FM row) on IV iii
  ii V (Gmaj9 F#m7 Em9 A13), and a short upright piano phrase on Dmaj9 as the end card comes
  up. Its last section starts exactly on the end card (`start`: odawg sizes the section
  before it and nudges its tempo to land there). Every part has a `role`.
- No `media/`: the mascot is `../art/logo.svg` (its mouth, eyes and waves are SVG ids the
  script shows and hides), the fonts are onimite's built-in DM Sans and JetBrains Mono, and
  every recorded sound is fetched on first use.

Mix (`onimite audio analyze demo/video.lua`): -14.9 LUFS integrated, -1.2 dBTP; the voices
at -14.1 LUFS, the score 10 LU under them (12 dB more while a line is said).

## Credits

- Voices: Pocket TTS by Kyutai (CC-BY-4.0), through ottis.
- Score samples: the virtuosity drum kit, Karoryfer Samples' growlybass, the Versilian
  Community Sample Library's upright piano (all CC0), fetched by odawg.
- Fonts: DM Sans and JetBrains Mono (SIL Open Font License 1.1), built into onimite.

Made with onimite; music by odawg.
