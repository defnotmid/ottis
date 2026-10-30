// Audio file reading (WAV, FLAC, MP3, Ogg Vorbis and Opus, MP4/M4A, WebM)
// folded to mono, and 16-bit PCM WAV output, through oicut.
package ottis_wav

import odaudio "deps:oicut/audio"
import odwav "deps:oicut/audio/wav"
import "core:fmt"
import "core:os"

Error :: Maybe(string)

// Decodes an audio file to mono f32 samples at the file's sample rate.
read :: proc(path: string, allocator := context.allocator) -> (samples: []f32, sample_rate: int, err: Error) {
	data, rerr := os.read_entire_file(path, context.temp_allocator)
	if rerr != nil do return nil, 0, fmt.aprintf("reading %s: %v", path, rerr)
	return decode(data, path, allocator)
}

// Decodes audio bytes in memory to mono f32 samples at the stream's sample
// rate. Several channels fold to their average, in place: the result is the
// front of the decoded buffer, and freeing it frees that buffer.
decode :: proc(data: []u8, label := "audio", allocator := context.allocator) -> (samples: []f32, sample_rate: int, err: Error) {
	if len(data) == 0 do return nil, 0, fmt.aprintf("%s: empty file", label)
	a, derr := odaudio.decode(data, allocator)
	if derr != .None || len(a.samples) == 0 || a.rate <= 0 || a.channels <= 0 {
		odaudio.destroy_audio(&a, allocator)
		return nil, 0, fmt.aprintf("%s: cannot decode audio (%v)", label, derr)
	}
	channels := a.channels
	if channels == 1 do return a.samples, a.rate, nil
	frames := len(a.samples) / channels
	// Frame i's channels start at i * channels >= i, so writing frame i
	// never overwrites a sample still to be read.
	for i in 0 ..< frames {
		sum: f32 = 0
		for s in a.samples[i * channels:][:channels] do sum += s
		a.samples[i] = sum / f32(channels)
	}
	return a.samples[:frames], a.rate, nil
}

// Writes mono f32 samples as a 16-bit PCM WAV file.
write_pcm16 :: proc(path: string, samples: []f32, sample_rate: int) -> odwav.Error {
	return odwav.write_file(path, samples, sample_rate, 1, .S16)
}
