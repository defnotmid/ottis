// Every downloaded file, pinned by URL revision, SHA-256 and size. A file
// is used only after its bytes match; nothing is fetched unpinned.
package assets

import "core:fmt"

Pin :: struct {
	path:   string, // under the data directory
	url:    string,
	sha256: string,
	size:   i64,
}

// ONNX Runtime 1.30.0 CPU release from github.com/microsoft/onnxruntime
// (MIT). Only the library and the licence files are kept from the archive.
ORT_VERSION :: "1.30.0"

Runtime_Release :: struct {
	archive: Pin,
	member:  string, // path of the library inside the archive, after the top directory
	library: string, // file name it is kept under
}

@(private)
ORT_URL :: "https://github.com/microsoft/onnxruntime/releases/download/v" + ORT_VERSION + "/"

when ODIN_OS == .Linux && ODIN_ARCH == .amd64 {
	RUNTIME :: Runtime_Release {
		archive = {"cache/onnxruntime-linux-x64-1.30.0.tgz", ORT_URL + "onnxruntime-linux-x64-1.30.0.tgz", "a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd", 11306877},
		member  = "lib/libonnxruntime.so.1.30.0",
		library = "libonnxruntime.so.1.30.0",
	}
} else when ODIN_OS == .Linux && ODIN_ARCH == .arm64 {
	RUNTIME :: Runtime_Release {
		archive = {"cache/onnxruntime-linux-aarch64-1.30.0.tgz", ORT_URL + "onnxruntime-linux-aarch64-1.30.0.tgz", "e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da", 10269495},
		member  = "lib/libonnxruntime.so.1.30.0",
		library = "libonnxruntime.so.1.30.0",
	}
} else when ODIN_OS == .Darwin && ODIN_ARCH == .arm64 {
	RUNTIME :: Runtime_Release {
		archive = {"cache/onnxruntime-osx-arm64-1.30.0.tgz", ORT_URL + "onnxruntime-osx-arm64-1.30.0.tgz", "6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012", 42373116},
		member  = "lib/libonnxruntime.1.30.0.dylib",
		library = "libonnxruntime.1.30.0.dylib",
	}
} else when ODIN_OS == .Windows && ODIN_ARCH == .amd64 {
	RUNTIME :: Runtime_Release {
		archive = {"cache/onnxruntime-win-x64-1.30.0.zip", ORT_URL + "onnxruntime-win-x64-1.30.0.zip", "c6ba983baf5681af108599675d2a89c2d145512d02de28aed0bff177cd0ba949", 82645522},
		member  = "lib/onnxruntime.dll",
		library = "onnxruntime.dll",
	}
} else {
	#panic("ottis: no pinned ONNX Runtime release for this platform")
}

// Pocket TTS english_2026-04 exported to ONNX by KevinAHM
// (huggingface.co/KevinAHM/pocket-tts-onnx, CC-BY-4.0, from Kyutai's
// weights), fp32 graphs.
BUNDLE_NAME :: "english_2026-04"

@(private)
BUNDLE_URL :: "https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/58a6d00cf13d239b6748cb0769f35c580a8f606c/onnx/"

BUNDLE := [?]Pin {
	{"english_2026-04/bundle.json", BUNDLE_URL + "english_2026-04/bundle.json", "bab643150f437f37df080a710520ff39ed9ebd9a339f8ebdc739f7eddfc28b3f", 24381},
	{"english_2026-04/tokenizer.model", BUNDLE_URL + "english_2026-04/tokenizer.model", "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6", 59339},
	{"english_2026-04/bos_before_voice.npy", BUNDLE_URL + "english_2026-04/bos_before_voice.npy", "f46edf4f7007b7ba4ea58831f49d003e59e167b4641c44bb3addfe9231a780b1", 4224},
	{"english_2026-04/text_conditioner.onnx", BUNDLE_URL + "english_2026-04/text_conditioner.onnx", "4ecee995fb69f85c7a7493d11f7b5ee15d9950facc7ab3f5c9c49ef1e03847bb", 16388344},
	{"english_2026-04/flow_lm_main.onnx", BUNDLE_URL + "english_2026-04/flow_lm_main.onnx", "6d18315e2c33ca3e3aa4a4e3dca22f56d007fd823127e24948b37695bf54190f", 302742149},
	{"english_2026-04/flow_lm_flow.onnx", BUNDLE_URL + "english_2026-04/flow_lm_flow.onnx", "085d239f68897e28fb06e95c743738ad8b8c092ee6dc55f5491313e81ff08062", 39097095},
	{"english_2026-04/mimi_decoder.onnx", BUNDLE_URL + "english_2026-04/mimi_decoder.onnx", "86f038caa02a96a0ff9c25526a0ff43a4906c418197ed72d3e30f720ac7ce802", 41471926},
	{"english_2026-04/mimi_encoder.onnx", BUNDLE_URL + "english_2026-04/mimi_encoder.onnx", "853e2ca623b8782d94c3745ec6133bfdff7ce33d9b11128bd29ea03f28d76e3d", 39768446},
	{"english_2026-04/LICENSE", BUNDLE_URL + "LICENSE", "fe7b4ce83b8381cc5b216bbb4af73c570688d1b819c73bbaed8ca401f4677cd6", 18655},
}

// The forced aligner that times words: wav2vec2-base-960h (Facebook AI,
// Apache-2.0; English letters, 20 ms frames) exported to ONNX by
// huggingface.co/onnx-community, fp32. The int8 export aligns as well
// but runs slower on CPU here.
ALIGNER :: Pin {
	"aligner/wav2vec2-base-960h.onnx",
	"https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX/resolve/729c1a6730fb549c20a1c73a3d3f96f11020225e/onnx/model.onnx",
	"00b7cc69516c1ab63c429e63a2b543e4d42bb77441ec5b98ee935de175b00de1",
	377911891,
}

// Kyutai's precomputed voices for english_2026-04: the backbone's KV cache
// after each voice prompt (huggingface.co/kyutai/pocket-tts-without-voice-cloning,
// CC-BY-4.0; the prompt recordings' own licences are listed at
// huggingface.co/kyutai/tts-voices). Fetched when first used.
Voice_Pin :: struct {
	name:   string,
	sha256: string,
	size:   i64,
}

@(private)
VOICE_URL :: "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/575d14372df89a4534bcb989870a34e77544b09f/languages/english_2026-04/embeddings/"

DEFAULT_VOICE :: "alba"

VOICES := [?]Voice_Pin {
	{"alba", "69c32db63ca56843d994f81f343f62e0bf2d73f7e4c9bc73e44bb1110b1d8845", 6194424},
	{"anna", "5ea82f78db006c9fd34e32ddd5aae82674b5b32646097977436458d00af80dfa", 7816440},
	{"azelma", "9f3e69f29075f991fd47774566865ef0e0e637cb5a35992c9919761b5b84b1de", 7963896},
	{"bill_boerst", "75610127d44e0b05b442154f80f89f993df235aecc6cad7070f11000d006c188", 6735096},
	{"caro_davy", "a5961b63a2e7a5cfd7edc383aa9042fb70fd14a9dee6310cdc633881a7f2449a", 5260536},
	{"charles", "299edc20182eeccfbf94e308626f259da4fbf339daa8d5905218f2b1774639b8", 6194424},
	{"cosette", "c4fdc15f5a3a20c44dd0064a37e87d15d25562936e8dbad7e07b9832015a545d", 6194424},
	{"daan", "a4d98b891a2850d19edd0f9bbcc6b550bcdac7a2ce29fe6be8a9110f915303d4", 4228912},
	{"eponine", "bda3b76a384ff355fe0350736387765946304ae8ca16e59f60ea3296a1c99cc6", 6931704},
	{"estelle", "ccebef7f51762c7fc08870f5ecf268e8713551ae2c9f7984ddaec0c1e1c77153", 8258808},
	{"eve", "ea9c2faf862a6c9d2cb61910fdf02842ae56940382cc8c1000fdb1b43269692b", 6538488},
	{"fantine", "51a8a4355d7f912d4959e4b1918314fda85ad47eba0a33a1d78a4a505d3465f5", 6538488},
	{"george", "0c1c6c57c55a98d81254b33728150c7776f40647fe95258d1a6c1a02780b5d02", 6243576},
	{"giovanni", "a5ee718157ec1c6fd9c1e66a7733c7e6337d474a74a7da756455d27717885c59", 4621552},
	{"jane", "37386227ca8ec5bf1b8e516c13d132ce5ff5437a304fe90129a1c62f41d9a008", 7374072},
	{"javert", "0ae88e03ca4e76a0e16cbf321a807428febda9d9e9bc0358c02e7f9c9e2c263b", 6194424},
	{"jean", "90be4b8f50bb4d2dbe27e3fb4e31417cf6a57928931f0a60426a1748821a3d12", 6194424},
	{"juergen", "e74d67b38339fc01118e3bbe2c90d6d40e601b161dbf7be906421956ba80a532", 6243576},
	{"lola", "34972e86b07b17272a8061460054a08119399cace9454979052b9e2d96664e8c", 5948664},
	{"marius", "04f84efcb77a0547ba582c058db496f7ff4920891d49d37b9950d128422582a8", 6194424},
	{"mary", "a8f2adf260cab966fe0a113d6b549d6efdeaa79de544ae0ff34b5b6a41445a59", 6194424},
	{"michael", "8937f724ac4719b9aa51ea0ba1f18f9de0af7a663ad6263558266c1a53c9722d", 7275768},
	{"paul", "ed7a019168f94dfe77009f1b0de59387abc6fbb0db954d38ce722ecb77da61aa", 6980856},
	{"peter_yearsley", "dd977a6e15591e347c9a23fa7cc09e35a65b462917f5eeb162baff6dc9e3f685", 3736816},
	{"rafael", "ac19f099f6cd839875a629c3e2e91e0dfc2c197acf2875db168d0ae244fb58bd", 6194424},
	{"stuart_bell", "5a49da7ca5df05d02587ec4a0981c0d318e045f68e24423c4203ce474d9b33dc", 5260536},
	{"vera", "4bf50ddd957b5d218b264fdcf18efbbc7384d12da3eca98ca19b9e8dd6976acc", 6735096},
}

// The pin for a named voice; its strings live in the temp allocator.
voice_pin :: proc(name: string) -> (Pin, bool) {
	for v in VOICES {
		if v.name == name {
			path := fmt.tprintf("%s/voices/%s.safetensors", BUNDLE_NAME, v.name)
			url := fmt.tprintf("%s%s.safetensors", VOICE_URL, v.name)
			return Pin{path, url, v.sha256, v.size}, true
		}
	}
	return {}, false
}
