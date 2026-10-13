#+build windows
package assets

import "core:fmt"
import "core:strings"
import win32 "core:sys/windows"

// WinHTTP ships with every Windows; Odin's core has no bindings for it, so
// these are the few calls a GET needs.
foreign import winhttp "system:winhttp.lib"

@(private)
HINTERNET :: rawptr

@(default_calling_convention = "system", private)
foreign winhttp {
	WinHttpOpen :: proc(agent: win32.LPCWSTR, access_type: win32.DWORD, proxy, bypass: win32.LPCWSTR, flags: win32.DWORD) -> HINTERNET ---
	WinHttpConnect :: proc(session: HINTERNET, server: win32.LPCWSTR, port: u16, reserved: win32.DWORD) -> HINTERNET ---
	WinHttpOpenRequest :: proc(connect: HINTERNET, verb, object, version, referrer: win32.LPCWSTR, accept_types: ^win32.LPCWSTR, flags: win32.DWORD) -> HINTERNET ---
	WinHttpSendRequest :: proc(request: HINTERNET, headers: win32.LPCWSTR, headers_length: win32.DWORD, optional: rawptr, optional_length, total_length: win32.DWORD, ctx: uintptr) -> win32.BOOL ---
	WinHttpReceiveResponse :: proc(request: HINTERNET, reserved: rawptr) -> win32.BOOL ---
	WinHttpQueryHeaders :: proc(request: HINTERNET, info_level: win32.DWORD, name: win32.LPCWSTR, buffer: rawptr, buffer_length: ^win32.DWORD, index: ^win32.DWORD) -> win32.BOOL ---
	WinHttpReadData :: proc(request: HINTERNET, buffer: rawptr, to_read: win32.DWORD, read: ^win32.DWORD) -> win32.BOOL ---
	WinHttpCloseHandle :: proc(h: HINTERNET) -> win32.BOOL ---
}

@(private) WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY :: 4
@(private) WINHTTP_FLAG_SECURE :: 0x00800000
@(private) WINHTTP_QUERY_STATUS_CODE :: 19
@(private) WINHTTP_QUERY_FLAG_NUMBER :: 0x20000000

// Streams `url` (https only; WinHTTP follows redirects) into `sink`.
http_get :: proc(url: string, sink: ^Sink) -> Error {
	if !strings.has_prefix(url, "https://") do return fmt.aprintf("only https URLs are fetched: %s", url)
	rest := url[len("https://"):]
	slash := strings.index_byte(rest, '/')
	host, object := rest, "/"
	if slash >= 0 do host, object = rest[:slash], rest[slash:]

	wide :: proc(s: string) -> win32.LPCWSTR {return win32.utf8_to_wstring(s, context.temp_allocator)}
	session := WinHttpOpen(wide("ottis"), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, nil, nil, 0)
	if session == nil do return fmt.aprintf("WinHttpOpen failed (%d)", win32.GetLastError())
	defer WinHttpCloseHandle(session)
	connect := WinHttpConnect(session, wide(host), 443, 0)
	if connect == nil do return fmt.aprintf("connecting to %s failed (%d)", host, win32.GetLastError())
	defer WinHttpCloseHandle(connect)
	request := WinHttpOpenRequest(connect, wide("GET"), wide(object), nil, nil, nil, WINHTTP_FLAG_SECURE)
	if request == nil do return fmt.aprintf("request for %s failed (%d)", url, win32.GetLastError())
	defer WinHttpCloseHandle(request)

	if !WinHttpSendRequest(request, nil, 0, nil, 0, 0, 0) || !WinHttpReceiveResponse(request, nil) {
		return fmt.aprintf("downloading %s failed (%d)", url, win32.GetLastError())
	}
	status: win32.DWORD
	status_size := win32.DWORD(size_of(status))
	if !WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, nil, &status, &status_size, nil) {
		return fmt.aprintf("reading the status of %s failed (%d)", url, win32.GetLastError())
	}
	if status != 200 do return fmt.aprintf("downloading %s: HTTP %d", url, status)

	buf: [64 * 1024]u8
	for {
		n: win32.DWORD
		if !WinHttpReadData(request, &buf[0], len(buf), &n) {
			return fmt.aprintf("reading %s failed (%d)", url, win32.GetLastError())
		}
		if n == 0 do break
		if err := sink_write(sink, buf[:n]); err != nil do return fmt.aprintf("writing %s: %v", url, err)
	}
	return nil
}
