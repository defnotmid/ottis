// Provisioning: the pinned ONNX Runtime library, the Pocket TTS bundle and
// voices and the word aligner, downloaded into one data directory and
// verified by SHA-256 before use.
//
// A verified file gets a `.sha256` stamp next to it; later runs trust a
// file whose size and stamp match its pin, and `fetch(verify = true)`
// hashes everything again.
package assets

import "core:crypto/sha2"
import "core:encoding/hex"
import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strings"

Error :: Maybe(string)

// The data directory: $OTTIS_DATA, else ottis/ under the user's data
// directory (~/.local/share, ~/Library/Application Support, %APPDATA%).
data_dir :: proc(allocator := context.allocator) -> (string, Error) {
	if env := os.get_env("OTTIS_DATA", allocator); env != "" do return env, nil
	base, err := os.user_data_dir(context.temp_allocator)
	if err != nil do return "", fmt.aprintf("no user data directory: %v", err)
	dir, _ := filepath.join({base, "ottis"}, allocator)
	return dir, nil
}

path_in :: proc(root, rel: string, allocator := context.temp_allocator) -> string {
	p, _ := filepath.join({root, rel}, allocator)
	return p
}

// Download destination: bytes go to a file and through SHA-256 together.
Sink :: struct {
	file:    ^os.File,
	hash:    sha2.Context_256,
	written: i64,
}

sink_write :: proc(s: ^Sink, data: []u8) -> os.Error {
	sha2.update(&s.hash, data)
	s.written += i64(len(data))
	_, err := os.write(s.file, data)
	return err
}

@(private)
hex_digest :: proc(ctx: ^sha2.Context_256) -> string {
	digest: [sha2.DIGEST_SIZE_256]u8
	sha2.final(ctx, digest[:])
	return string(hex.encode(digest[:], context.temp_allocator))
}

@(private)
hash_file :: proc(path: string) -> (string, Error) {
	f, err := os.open(path)
	if err != nil do return "", fmt.aprintf("opening %s: %v", path, err)
	defer os.close(f)
	ctx: sha2.Context_256
	sha2.init_256(&ctx)
	buf := make([]u8, 1 << 20, context.temp_allocator)
	for {
		n, rerr := os.read(f, buf)
		if n > 0 do sha2.update(&ctx, buf[:n])
		if rerr == .EOF || n == 0 do break
		if rerr != nil do return "", fmt.aprintf("reading %s: %v", path, rerr)
	}
	return hex_digest(&ctx), nil
}

@(private)
make_dirs :: proc(dir: string) -> Error {
	err := os.make_directory_all(dir)
	if err != nil && err != .Exist do return fmt.aprintf("creating %s: %v", dir, err)
	return nil
}

@(private)
stamp_path :: proc(path: string) -> string {
	return strings.concatenate({path, ".sha256"}, context.temp_allocator)
}

// Whether `path` holds `sha256` (and `size`, when known): by its stamp, or
// by hashing when `verify` is set or the stamp is missing.
@(private)
holds :: proc(path, sha256: string, size: i64, verify: bool) -> bool {
	info, err := os.stat(path, context.temp_allocator)
	if err != nil do return false
	if size > 0 && info.size != size do return false
	if !verify {
		stamp, serr := os.read_entire_file(stamp_path(path), context.temp_allocator)
		if serr == nil && strings.trim_space(string(stamp)) == sha256 do return true
	}
	got, herr := hash_file(path)
	if herr != nil || got != sha256 do return false
	_ = os.write_entire_file(stamp_path(path), sha256)
	return true
}

Progress :: proc(message: string)

@(private)
say :: proc(progress: Progress, format: string, args: ..any) {
	if progress != nil do progress(fmt.tprintf(format, ..args))
}

// Makes `root/pin.path` hold the pinned bytes, downloading when needed.
ensure :: proc(root: string, pin: Pin, verify := false, progress: Progress = nil) -> Error {
	path := path_in(root, pin.path)
	if holds(path, pin.sha256, pin.size, verify) do return nil

	make_dirs(filepath.dir(path)) or_return
	say(progress, "fetching %s (%.1f MB)", pin.path, f64(pin.size) / 1e6)
	part := strings.concatenate({path, ".part"}, context.temp_allocator)
	f, err := os.create(part)
	if err != nil do return fmt.aprintf("creating %s: %v", part, err)
	sink := Sink{file = f}
	sha2.init_256(&sink.hash)
	get_err := http_get(pin.url, &sink)
	os.close(f)
	if get_err != nil {
		os.remove(part)
		return get_err
	}
	got := hex_digest(&sink.hash)
	if got != pin.sha256 || sink.written != pin.size {
		os.remove(part)
		return fmt.aprintf("%s: got %d bytes with SHA-256 %s, pinned %d bytes with %s", pin.url, sink.written, got, pin.size, pin.sha256)
	}
	os.remove(path)
	if rerr := os.rename(part, path); rerr != nil do return fmt.aprintf("renaming %s: %v", part, rerr)
	_ = os.write_entire_file(stamp_path(path), pin.sha256)
	return nil
}

RUNTIME_DIR :: "onnxruntime-" + ORT_VERSION

// The ONNX Runtime library for this platform, extracted from the pinned
// release with its LICENSE and ThirdPartyNotices.txt. Returns its path.
ensure_runtime :: proc(root: string, verify := false, progress: Progress = nil) -> (library: string, err: Error) {
	dir := path_in(root, RUNTIME_DIR)
	library = path_in(dir, RUNTIME.library, context.allocator)
	names := [3]string{RUNTIME.library, "LICENSE", "ThirdPartyNotices.txt"}
	members := [3]string{RUNTIME.member, "LICENSE", "ThirdPartyNotices.txt"}
	present := true
	for name in names do present &&= os.exists(path_in(dir, name))
	if present && !verify do return library, nil

	ensure(root, RUNTIME.archive, verify, progress) or_return
	if present do return library, nil
	say(progress, "extracting %s", RUNTIME.library)
	archive, rerr := os.read_entire_file(path_in(root, RUNTIME.archive.path), context.allocator)
	if rerr != nil do return library, fmt.aprintf("reading %s: %v", RUNTIME.archive.path, rerr)
	defer delete(archive)
	make_dirs(dir) or_return
	files := extract(archive, members[:], strings.has_suffix(RUNTIME.archive.path, ".zip")) or_return
	defer {
		for f in files do delete(f)
		delete(files)
	}
	for data, i in files {
		path := path_in(dir, names[i])
		part := strings.concatenate({path, ".part"}, context.temp_allocator)
		if werr := os.write_entire_file(part, data, os.Permissions_Read_All + {.Write_User, .Execute_User}); werr != nil {
			return library, fmt.aprintf("writing %s: %v", part, werr)
		}
		os.remove(path)
		if werr := os.rename(part, path); werr != nil do return library, fmt.aprintf("renaming %s: %v", part, werr)
	}
	return library, nil
}

ensure_bundle :: proc(root: string, verify := false, progress: Progress = nil) -> Error {
	for pin in BUNDLE do ensure(root, pin, verify, progress) or_return
	return ensure(root, ALIGNER, verify, progress)
}

ensure_voice :: proc(root, name: string, verify := false, progress: Progress = nil) -> Error {
	pin, ok := voice_pin(name)
	if !ok {
		names := make([dynamic]string, context.temp_allocator)
		for v in VOICES do append(&names, v.name)
		return fmt.aprintf("unknown voice %q; voices: %s", name, strings.join(names[:], ", ", context.temp_allocator))
	}
	return ensure(root, pin, verify, progress)
}

// Everything `say` needs for `voice`.
fetch :: proc(root, voice: string, verify := false, progress: Progress = nil) -> (library: string, err: Error) {
	library = ensure_runtime(root, verify, progress) or_return
	ensure_bundle(root, verify, progress) or_return
	ensure_voice(root, voice, verify, progress) or_return
	return library, nil
}
