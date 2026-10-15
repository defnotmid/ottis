// Audio file reading (WAV, FLAC, MP3, Ogg Vorbis) and 16-bit PCM WAV
// output, through vendor:miniaudio.
package ottis_wav

import "core:fmt"
import "core:io"
import "core:os"
import ma "vendor:miniaudio"

Error :: Maybe(string)

// Decodes an audio file (WAV, FLAC, MP3, Ogg Vorbis) to mono f32 samples
// at the file's sample rate.
read :: proc(path: string, allocator := context.allocator) -> (samples: []f32, sample_rate: int, err: Error) {
	data, rerr := os.read_entire_file(path, context.temp_allocator)
	if rerr != nil do return nil, 0, fmt.aprintf("reading %s: %v", path, rerr)
	return decode(data, path, allocator)
}

// Decodes audio bytes in memory to mono f32 samples at the stream's sample rate.
decode :: proc(data: []u8, label := "audio", allocator := context.allocator) -> (samples: []f32, sample_rate: int, err: Error) {
	if len(data) == 0 do return nil, 0, fmt.aprintf("%s: empty file", label)
	cfg := ma.decoder_config_init(.f32, 1, 0)
	frames: u64
	pcm: rawptr
	if res := ma.decode_memory(raw_data(data), len(data), &cfg, &frames, &pcm); res != .SUCCESS || pcm == nil || frames == 0 || cfg.sampleRate == 0 {
		return nil, 0, fmt.aprintf("%s: cannot decode audio (%v)", label, res)
	}
	defer ma.free(pcm, nil)
	samples = make([]f32, int(frames), allocator)
	copy(samples, ([^]f32)(pcm)[:int(frames)])
	return samples, int(cfg.sampleRate), nil
}

// Writes mono f32 samples as a 16-bit PCM WAV file.
write_pcm16 :: proc(path: string, samples: []f32, sample_rate: int) -> os.Error {
	Write_Ctx :: struct {
		file: ^os.File,
		err:  os.Error,
	}
	on_write :: proc "c" (enc: ^ma.encoder, buf: rawptr, bytes_to_write: uint, bytes_written: ^uint) -> ma.result {
		ctx := (^Write_Ctx)(enc.pUserData)
		context = {}
		n, err := os.write(ctx.file, ([^]u8)(buf)[:bytes_to_write])
		if err != nil {
			ctx.err = err
			return .ERROR
		}
		if bytes_written != nil do bytes_written^ = uint(n)
		return .SUCCESS
	}
	on_seek :: proc "c" (enc: ^ma.encoder, offset: i64, origin: ma.seek_origin) -> ma.result {
		ctx := (^Write_Ctx)(enc.pUserData)
		context = {}
		whence: io.Seek_From
		switch origin {
		case .start:   whence = .Start
		case .current: whence = .Current
		case .end:     whence = .End
		}
		if _, err := os.seek(ctx.file, offset, whence); err != nil {
			ctx.err = err
			return .ERROR
		}
		return .SUCCESS
	}

	f, oerr := os.open(path, {.Write, .Create, .Trunc}, os.Permissions_Read_All + {.Write_User})
	if oerr != nil do return oerr
	defer os.close(f)

	wctx := Write_Ctx{file = f}
	cfg := ma.encoder_config_init(.wav, .s16, 1, u32(sample_rate))
	enc: ma.encoder
	if ma.encoder_init(on_write, on_seek, &wctx, &cfg, &enc) != .SUCCESS {
		return wctx.err if wctx.err != nil else .Invalid_Argument
	}
	defer ma.encoder_uninit(&enc)

	if len(samples) > 0 {
		converter_cfg := ma.data_converter_config_init(.f32, .s16, 1, 1, u32(sample_rate), u32(sample_rate))
		converter: ma.data_converter
		if ma.data_converter_init(&converter_cfg, nil, &converter) != .SUCCESS do return .Invalid_Argument
		defer ma.data_converter_uninit(&converter, nil)

		s16_buf := make([]i16, len(samples), context.temp_allocator)
		in_frames := u64(len(samples))
		out_frames := u64(len(samples))
		ma.data_converter_process_pcm_frames(&converter, raw_data(samples), &in_frames, raw_data(s16_buf), &out_frames)
		written: u64
		if ma.encoder_write_pcm_frames(&enc, raw_data(s16_buf), out_frames, &written) != .SUCCESS {
			return wctx.err if wctx.err != nil else .Invalid_Argument
		}
	}
	return wctx.err
}

