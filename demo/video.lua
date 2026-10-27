-- ottis's demo (defnotmid.com/ottis): Ottis, the mascot, says what ottis does
-- in four of its voices. Every line is spoken by ottis through onimite's
-- say(), and Ottis's mouth, the captions and the word table all read the word
-- times it returns. Under the voice, a score (track.lua: jazz-hop, odawg)
-- ducks on every line and resolves as the end card comes up.
--
--     onimite render demo/video.lua --qp 22 -o demo/demo.mp4    # from ottis's root
--
-- No media of its own: the mascot is ../art/logo.svg, the fonts are onimite's
-- built-in ones, ottis downloads its model and voices on first use, and odawg
-- fetches the score's sample sets on first use.
-- Voices: Pocket TTS by Kyutai (CC-BY-4.0).

local W, H = 1920, 1080
local BG, PANEL, INK, DIM, TEAL, SAND = "#0b1622", "#122233", "#e2e8f0", "#64748b", "#5eead4", "#fbbf24"
-- The mascot, read once: an SVG outside media/ is drawn from its markup.
local file = assert(io.open(SCRIPT_DIR .. "/../art/logo.svg"), "ottis's art/logo.svg is missing")
local LOGO = file:read("a")
file:close()

-- The lines in the order they're said, each in its voice.
local LINES = {
  { "hello", "Hi, I'm Ottis. I turn text into speech, right on your computer.", voice = "alba", seed = 1 },
  { "times", "And I tell you when every word is said.", voice = "alba", seed = 2 },
  { "fetch", "Each voice is a few megabytes, fetched the first time you use it.", voice = "marius", seed = 1 },
  { "speed", "It all runs on the CPU, faster than real time.", voice = "eve", seed = 1 },
  { "small", "No Python, nothing to install: one small binary.", voice = "george", seed = 1 },
  { "close", "Captions, lip sync and cues all follow my words.", voice = "alba", seed = 3 },
}
local LEAD, GAP, TAIL = 0.8, 0.5, 3.2   -- before the first line, between lines, the end card

-- Say them, and lay them out end to end from their lengths.
local at = LEAD
for _, l in ipairs(LINES) do
  l.line = say(l[1], l[2], { voice = l.voice, seed = l.seed })
  l.at = at
  at = at + l.line.duration + GAP
end
local END = at - GAP + 0.4               -- the end card's start
local DURATION = END + TAIL
-- The score: its last section starts on the end card.
local score = require("track")(END)

-- The mouth, from the words: within a word one shape per vowel group (a: open,
-- e i y: ee, o u w: o), each held for its share of the word and closing
-- briefly before the next. Built once here: render only looks a key up.
local SHAPE = { a = "mouth_open", e = "mouth_ee", i = "mouth_ee", y = "mouth_ee", o = "mouth_o", u = "mouth_o", w = "mouth_o" }
local mouth = {}                         -- { start, stop, shape } on the video clock, in order
for _, l in ipairs(LINES) do
  for _, w in ipairs(l.line.words) do
    local shapes = {}
    for v in w.text:lower():gmatch("[aeiouyw]+") do shapes[#shapes + 1] = SHAPE[v:sub(1, 1)] end
    if #shapes == 0 then shapes[1] = "mouth_open" end
    local t0, span = l.at + w.start, (w.stop - w.start) / #shapes
    for k, s in ipairs(shapes) do
      mouth[#mouth + 1] = { t0 + (k - 1) * span, t0 + (k - 0.25) * span, s }
    end
  end
end

-- The line being said at t (or the last one said, until the next starts).
local function line_at(t)
  local cur
  for _, l in ipairs(LINES) do
    if t >= l.at - 0.2 then cur = l end
  end
  return cur
end

local function mouth_at(t)
  for _, m in ipairs(mouth) do
    if t < m[1] then return "mouth_closed" end
    if t < m[2] then return m[3] end
  end
  return "mouth_closed"
end

local MOUTHS = { "mouth_closed", "mouth_open", "mouth_o", "mouth_ee" }

local function ottis(t, x, y, size)
  local m = mouth_at(t)
  local talking = m ~= "mouth_closed"
  local blink = (t % 3.7) < 0.12
  local o = {
    eyes_open = { visible = not blink }, eyes_closed = { visible = blink },
    waves_a = { visible = talking and math.floor(t * 6) % 2 == 0 },
    waves_b = { visible = talking and math.floor(t * 6) % 2 == 1 },
  }
  for _, id in ipairs(MOUTHS) do o[id] = { visible = id == m } end
  svg { markup = LOGO, x = x, y = y, w = size, h = size, overrides = o }
end

-- The word table: the line's words and the times ottis gave them, the word
-- being said lit (through the pause after it, as the captions do), the ones
-- to come dim.
local function word_table(t, l, x, y)
  local lt = t - l.at
  local words = l.line.words
  text { "voice: " .. l.voice, x = x, y = y, font = "JetBrains Mono", size = 30, fill = TEAL }
  text { "word", x = x, y = y + 70, font = "JetBrains Mono", size = 24, fill = DIM }
  text { "start", x = x + 470, y = y + 70, align = "right", font = "JetBrains Mono", size = 24, fill = DIM }
  text { "stop", x = x + 640, y = y + 70, align = "right", font = "JetBrains Mono", size = 24, fill = DIM }
  for i, w in ipairs(words) do
    local ry = y + 70 + i * 50
    local next_start = words[i + 1] and words[i + 1].start or l.line.duration + 0.3
    local now = lt >= w.start and lt < next_start
    local said = lt >= next_start
    local fill = now and SAND or (said and INK or DIM)
    local a = animate(t, { { start = l.at + w.start - 0.15, stop = l.at + w.start, from = 0.35, to = 1 } })
    if now then rect { x = x - 16, y = ry - 36, w = 672, h = 48, rx = 8, fill = "#fbbf2422" } end
    group({ opacity = a }, function()
      text { w.text, x = x, y = ry, font = "DM Sans", weight = 500, size = 34, fill = fill }
      text { string.format("%.2f", w.start), x = x + 470, y = ry, align = "right", font = "JetBrains Mono", size = 30, fill = fill }
      text { string.format("%.2f", w.stop), x = x + 640, y = ry, align = "right", font = "JetBrains Mono", size = 30, fill = fill }
    end)
  end
end

video { width = W, height = H, fps = 60, duration = DURATION, background = BG,
  -- the lines on the voice bus (a say line's own), the score under them,
  -- ducked on every line
  buses = { voice = {}, music = { gain_db = 1.6 } },  -- the score 10 LU under the voices
  master_gain_db = 6.5,   -- ottis says every line at -23 LUFS: lifted to about -14.5
  audio = (function()
    local tracks = { { "score", bus = "music", ducked = true, fade_out = 1.5 } }
    for _, l in ipairs(LINES) do tracks[#tracks + 1] = { l[1], start = l.at } end
    return tracks
  end)(),
  render = function(f)
    local t = f.seconds
    local out = animate(t, { { start = END - 0.3, stop = END + 0.3, from = 1, to = 0, easing = ease_in_out } })

    -- Ottis on the left, on a soft glow; the title above.
    circle { cx = 520, cy = 470, r = 380, fill = radial_gradient { cx = 520, cy = 470, r = 380,
      stops = { { 0, "#5eead41f" }, { 1, "#5eead400" } } } }
    ottis(t, 200, 150, 640)
    group({ opacity = out }, function()
      text { "ottis", x = 120, y = 112, font = "DM Sans", weight = 700, size = 80, fill = INK }
      text { "text to speech with word timings, in Odin", x = 340, y = 108, font = "DM Sans", size = 34, fill = DIM }

      -- The words and their times on the right.
      local l = line_at(t)
      if l then
        rect { x = 1080, y = 150, w = 720, h = 860, rx = 24, fill = PANEL }
        word_table(t, l, 1120, 215)
      end

      -- The captions under Ottis.
      for _, l in ipairs(LINES) do
        captions(f, { src = l[1], start = l.at, x = 120, y = 930, w = 880, valign = "middle",
          font = "DM Sans", weight = 500, size = 46, line_height = 1.2, fill = INK, highlight = "fill",
          highlight_fill = SAND, upcoming_fill = DIM, max_duration = 2.2, hold = 0.3 })
      end
    end)

    -- The end card.
    local card = animate(t, { { start = END, stop = END + 0.6, from = 0, to = 1, easing = ease_out } })
    group({ opacity = card }, function()
      text { "ottis", x = 1440, y = 470, align = "center", font = "DM Sans", weight = 700, size = 200, fill = INK }
      text { "text to speech with word timings, in Odin", x = 1440, y = 560, align = "center", font = "DM Sans", size = 40, fill = TEAL }
      text { "github.com/defnotmid/ottis  ·  MIT", x = 1440, y = 650, align = "center", font = "JetBrains Mono", size = 30, fill = DIM }
      text { "Voices: Pocket TTS by Kyutai (CC-BY-4.0)", x = 1440, y = 700, align = "center", font = "DM Sans", size = 28, fill = DIM }
    end)
  end }
