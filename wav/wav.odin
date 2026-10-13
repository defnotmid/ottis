// Mono 16-bit PCM WAV output.
package wav

import "core:encoding/endian"
import "core:math"
import "core:os"

write_pcm16 :: proc(path: string, samples: []f32, sample_rate: int) -> os.Error {
	data_bytes := len(samples) * 2
	out := make([]u8, 44 + data_bytes)
	defer delete(out)
	copy(out[0:], "RIFF")
	endian.unchecked_put_u32le(out[4:], u32(36 + data_bytes))
	copy(out[8:], "WAVEfmt ")
	endian.unchecked_put_u32le(out[16:], 16)
	endian.unchecked_put_u16le(out[20:], 1) // PCM
	endian.unchecked_put_u16le(out[22:], 1) // mono
	endian.unchecked_put_u32le(out[24:], u32(sample_rate))
	endian.unchecked_put_u32le(out[28:], u32(sample_rate * 2))
	endian.unchecked_put_u16le(out[32:], 2)
	endian.unchecked_put_u16le(out[34:], 16)
	copy(out[36:], "data")
	endian.unchecked_put_u32le(out[40:], u32(data_bytes))
	for s, i in samples {
		v := math.round(clamp(s, -1, 1) * 32767)
		endian.unchecked_put_u16le(out[44 + i * 2:], u16(i16(v)))
	}
	return os.write_entire_file(path, out)
}
