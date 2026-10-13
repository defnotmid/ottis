#+build !windows
package assets

import "base:runtime"
import "core:c"
import "core:fmt"
import "core:os"
import "core:strings"
import curl "vendor:curl"

@(private)
Curl_Sink :: struct {
	ctx:  runtime.Context,
	sink: ^Sink,
	err:  os.Error,
}

@(private)
curl_write :: proc "c" (data: [^]u8, size, count: c.size_t, user: rawptr) -> c.size_t {
	s := (^Curl_Sink)(user)
	context = s.ctx
	n := int(size * count)
	if err := sink_write(s.sink, data[:n]); err != nil {
		s.err = err
		return 0
	}
	return c.size_t(n)
}

// Streams `url` (redirects followed, HTTP errors are failures) into `sink`.
http_get :: proc(url: string, sink: ^Sink) -> Error {
	@(static) initialized: bool
	if !initialized {
		curl.global_init(curl.GLOBAL_DEFAULT)
		initialized = true
	}
	h := curl.easy_init()
	if h == nil do return fmt.aprintf("curl: cannot create a handle")
	defer curl.easy_cleanup(h)

	cs := Curl_Sink{ctx = context, sink = sink}
	curl.easy_setopt(h, .URL, strings.clone_to_cstring(url, context.temp_allocator))
	curl.easy_setopt(h, .FOLLOWLOCATION, c.long(1))
	curl.easy_setopt(h, .FAILONERROR, c.long(1))
	curl.easy_setopt(h, .USERAGENT, cstring("ottis"))
	curl.easy_setopt(h, .WRITEFUNCTION, curl_write)
	curl.easy_setopt(h, .WRITEDATA, rawptr(&cs))
	code := curl.easy_perform(h)
	if cs.err != nil do return fmt.aprintf("writing %s: %v", url, cs.err)
	if code != .E_OK do return fmt.aprintf("downloading %s: %s", url, string(curl.easy_strerror(code)))
	return nil
}
