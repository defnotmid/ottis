-- The score: an odawg song under Ottis's voice, jazz-hop after Nujabes at
-- 86 BPM in D major. It is a bed, not a tune (the voice is the melody): a
-- swung boom-bap on a real kit, an upright bass under it, an electric piano
-- of our own on IV iii ii V (Gmaj9 F#m7 Em9 A13), resolving to Dmaj9 with a
-- short piano phrase as the end card comes up. The video ducks it under each
-- line; its last section starts exactly where the end card does (`start`:
-- odawg sizes the section before it and nudges its tempo to land there).
-- Every recorded sound is a sample set odawg fetches by name (virtuosity's
-- kit, Karoryfer's growlybass, VCSL's upright piano), so the demo needs no
-- media of its own.

return function(card_at)
  return song { "score", key = "D major", bpm = 86, seed = 86, swing = 0.22, swing_subdivision = "16n",
    instruments = { {
      -- the electric piano: a sine carrier, a sine modulator at its pitch
      -- whose depth falls with the note and a high one for the tine's knock,
      -- a slow decay to a soft sustain (the shape of onimite's library
      -- `keys`, written here)
      rhodes = {
        nodes = {
          menv = { "Decay", time = 0.9 },
          mod = { "Osc", wave = "sine", ratio = 1, amp = { "menv", "velocity" } },
          tenv = { "Decay", time = 0.025 },
          tine = { "Osc", wave = "sine", ratio = 14, amp = { "tenv", "velocity" } },
          carrier = { "Osc", wave = "sine", frequency = { { by = "mod", index = 1.4 }, { by = "tine", index = 0.3 } } },
          env = { "Env", attack = 0.002, decay = 2.4, sustain = 0.25, release = 0.5 },
          out = { "Mix", "carrier", amp = "env", gain_db = -8.4 },
        },
        out = "out",
      },
    } },
    sections = {
      { "talk", duration = "8m", flex = { "4m", "16m" }, chords = "Gmaj9 F#m7 Em9 A13" },
      { "card", start = card_at, duration = "2m", chords = "Dmaj9" },
    },
    returns = { room = { { "Reverb", decay = 1.2, pre_delay = 0.01 }, { "Filter", type = "highpass", frequency = 250 } } },
    parts = {
      -- the kit: a lazy boom-bap, the snare on two and four, ghosts before
      -- them, swung hats; a soft cymbal and the last kick on the card
      drums = { role = "drums", instrument = { "Sampler", set = "virtuosity/drums" }, gain_db = 0, sends = { room = -16 }, play = {
        { "talk", humanize = 0.006, velocity = 0.75, steps = {
          kick = { "x.......x.x.....", "x......x..x....." }, snare = { "....x.......x...", "....x..o....x..o" },
          hat = "x.o.x.o.x.o.x.o." } },
        { "card", events = { { time = 0, note = "kick", velocity = 1 }, { time = 0, note = "crash", velocity = 0.4 } } },
      } },
      -- the bass: the root, a walk up to the next chord
      bass = { role = "bass", instrument = { "Sampler", set = "karoryfer/growlybass", release = 0.3 }, gain_db = -9, play = {
        { "talk", degrees = "1 - - - - - - . 5 - . 3 - - 2 .", subdivision = "16n", octave = 2, velocity = 0.7, humanize = 0.004 },
        { "card", degrees = "1", subdivision = "1m", octave = 2, velocity = 0.6 },
      } },
      -- the electric piano: the chord laid back, behind the beat
      keys = { role = "comp", instrument = "rhodes", gain_db = 0, sends = { room = -10 },
        effects = { { "Chorus", frequency = 0.8, delay_time = 0.01, depth = 0.4, wet = 0.3 } }, play = {
        { "talk", chords = { voicing = "spread", rhythm = { "x.........x.....", "...x......x....." } }, octave = 3, duration = "4n", velocity = 0.55, humanize = 0.01 },
        { "card", chords = { voicing = "spread", rhythm = "x..............." }, octave = 3, duration = "2m", velocity = 0.28 },
      } },
      -- the card's phrase: the upright up the Dmaj9 and home
      piano = { role = "melody", instrument = { "Sampler", set = "vcsl/upright_piano", release = 1.5 }, gain_db = -3, sends = { room = -8 }, play = {
        { "card", notes = ". F#5 A5 C#6 E6 - C#6 A5 | D6 - - - - - - -", subdivision = "8n", velocity = 0.5, humanize = 0.01 },
      } },
    },
  }
end
