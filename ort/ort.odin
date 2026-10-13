// ONNX Runtime through its C API, loaded at run time from the library
// `ottis fetch` placed, so building ottis needs no ONNX Runtime and the
// process links none. The API is one function table reached through
// OrtGetApiBase; the slots below are API 27's (unchanged through 30),
// as in ../dangersnail/extras/onnx_runtime/bindings.odin.
//
// Values are ORT's own: a session's outputs are allocated by ORT and can be
// fed straight back as the next run's inputs (the model's state), and
// inputs made from caller memory borrow it without copying.
package ort

import "core:c"
import "core:dynlib"
import "core:fmt"
import "core:os"
import "core:strings"

API_VERSION :: 27

Status :: struct {}
Env :: struct {}
Session_Options :: struct {}
Session :: struct {}
Value :: struct {}
Memory_Info :: struct {}
Allocator :: struct {}
Tensor_Info :: struct {}

Element_Type :: enum c.int {
	Undefined = 0,
	Float32   = 1,
	Uint8     = 2,
	Int8      = 3,
	Int64     = 7,
	Bool      = 9,
	Float16   = 10,
}

@(private)
Api_Base :: struct {
	get_api:            proc "c" (version: u32) -> [^]rawptr,
	get_version_string: proc "c" () -> cstring,
}

@(private)
SLOT :: enum {
	Get_Error_Message                    = 2,
	Create_Env                           = 3,
	Create_Session                       = 7,
	Run                                  = 9,
	Create_Session_Options               = 10,
	Set_Session_Execution_Mode           = 13,
	Set_Session_Graph_Optimization_Level = 23,
	Set_Intra_Op_Num_Threads             = 24,
	Set_Inter_Op_Num_Threads             = 25,
	Session_Get_Input_Count              = 30,
	Session_Get_Output_Count             = 31,
	Session_Get_Input_Name               = 36,
	Session_Get_Output_Name              = 37,
	Create_Tensor_With_Data              = 49,
	Get_Tensor_Mutable_Data              = 51,
	Get_Tensor_Element_Type              = 60,
	Get_Dimensions_Count                 = 61,
	Get_Dimensions                       = 62,
	Get_Tensor_Type_And_Shape            = 65,
	Create_Cpu_Memory_Info               = 69,
	Allocator_Free                       = 76,
	Get_Allocator_With_Default_Options   = 78,
	Release_Env                          = 92,
	Release_Status                       = 93,
	Release_Memory_Info                  = 94,
	Release_Session                      = 95,
	Release_Value                        = 96,
	Release_Tensor_Type_And_Shape_Info   = 99,
	Release_Session_Options              = 100,
}

// The procedures ottis calls, typed once from the table.
Runtime :: struct {
	library:         dynlib.Library,
	version:         string,
	env:             ^Env,
	memory:          ^Memory_Info,
	allocator:       ^Allocator,
	error_message:   proc "c" (s: ^Status) -> cstring,
	create_env:      proc "c" (level: c.int, id: cstring, out: ^^Env) -> ^Status,
	create_session:  proc "c" (env: ^Env, path: rawptr, options: ^Session_Options, out: ^^Session) -> ^Status,
	run:             proc "c" (s: ^Session, options: rawptr, input_names: [^]cstring, inputs: [^]^Value, input_count: c.size_t, output_names: [^]cstring, output_count: c.size_t, outputs: [^]^Value) -> ^Status,
	create_options:  proc "c" (out: ^^Session_Options) -> ^Status,
	set_exec_mode:   proc "c" (o: ^Session_Options, mode: c.int) -> ^Status,
	set_opt_level:   proc "c" (o: ^Session_Options, level: c.int) -> ^Status,
	set_intra:       proc "c" (o: ^Session_Options, n: c.int) -> ^Status,
	set_inter:       proc "c" (o: ^Session_Options, n: c.int) -> ^Status,
	input_count:     proc "c" (s: ^Session, out: ^c.size_t) -> ^Status,
	output_count:    proc "c" (s: ^Session, out: ^c.size_t) -> ^Status,
	input_name:      proc "c" (s: ^Session, i: c.size_t, a: ^Allocator, out: ^cstring) -> ^Status,
	output_name:     proc "c" (s: ^Session, i: c.size_t, a: ^Allocator, out: ^cstring) -> ^Status,
	create_tensor:   proc "c" (m: ^Memory_Info, data: rawptr, bytes: c.size_t, shape: [^]i64, rank: c.size_t, type: Element_Type, out: ^^Value) -> ^Status,
	tensor_data:     proc "c" (v: ^Value, out: ^rawptr) -> ^Status,
	element_type:    proc "c" (info: ^Tensor_Info, out: ^Element_Type) -> ^Status,
	dims_count:      proc "c" (info: ^Tensor_Info, out: ^c.size_t) -> ^Status,
	dims:            proc "c" (info: ^Tensor_Info, out: [^]i64, n: c.size_t) -> ^Status,
	type_and_shape:  proc "c" (v: ^Value, out: ^^Tensor_Info) -> ^Status,
	create_cpu_info: proc "c" (alloc_type: c.int, mem_type: c.int, out: ^^Memory_Info) -> ^Status,
	allocator_free:  proc "c" (a: ^Allocator, p: rawptr) -> ^Status,
	default_alloc:   proc "c" (out: ^^Allocator) -> ^Status,
	release_env:     proc "c" (p: ^Env),
	release_status:  proc "c" (p: ^Status),
	release_memory:  proc "c" (p: ^Memory_Info),
	release_session: proc "c" (p: ^Session),
	release_value:   proc "c" (p: ^Value),
	release_info:    proc "c" (p: ^Tensor_Info),
	release_options: proc "c" (p: ^Session_Options),
}

// Errors carry ORT's message; the caller owns the string.
Error :: Maybe(string)

@(private)
check :: proc(rt: ^Runtime, s: ^Status, what: string) -> Error {
	if s == nil do return nil
	message := fmt.aprintf("%s: %s", what, string(rt.error_message(s)))
	rt.release_status(s)
	return message
}

init :: proc(rt: ^Runtime, library_path: string) -> Error {
	// A hardened-runtime process (macOS, as a notarized app embedding ottis
	// is) refuses to dlopen a relative path, and a data dir may be one.
	path, aerr := os.get_absolute_path(library_path, context.temp_allocator)
	if aerr != nil do path = library_path
	library, ok := dynlib.load_library(path)
	if !ok do return fmt.aprintf("loading %s: %s", library_path, dynlib.last_error())
	rt.library = library
	symbol, found := dynlib.symbol_address(library, "OrtGetApiBase")
	if !found do return fmt.aprintf("%s has no OrtGetApiBase", library_path)
	base := (proc "c" () -> ^Api_Base)(symbol)()
	rt.version = string(base.get_version_string())
	api := base.get_api(API_VERSION)
	if api == nil do return fmt.aprintf("ONNX Runtime %s does not provide C API %d", rt.version, API_VERSION)

	slot :: proc(api: [^]rawptr, s: SLOT, $T: typeid) -> T {return T(api[int(s)])}
	rt.error_message = slot(api, .Get_Error_Message, type_of(rt.error_message))
	rt.create_env = slot(api, .Create_Env, type_of(rt.create_env))
	rt.create_session = slot(api, .Create_Session, type_of(rt.create_session))
	rt.run = slot(api, .Run, type_of(rt.run))
	rt.create_options = slot(api, .Create_Session_Options, type_of(rt.create_options))
	rt.set_exec_mode = slot(api, .Set_Session_Execution_Mode, type_of(rt.set_exec_mode))
	rt.set_opt_level = slot(api, .Set_Session_Graph_Optimization_Level, type_of(rt.set_opt_level))
	rt.set_intra = slot(api, .Set_Intra_Op_Num_Threads, type_of(rt.set_intra))
	rt.set_inter = slot(api, .Set_Inter_Op_Num_Threads, type_of(rt.set_inter))
	rt.input_count = slot(api, .Session_Get_Input_Count, type_of(rt.input_count))
	rt.output_count = slot(api, .Session_Get_Output_Count, type_of(rt.output_count))
	rt.input_name = slot(api, .Session_Get_Input_Name, type_of(rt.input_name))
	rt.output_name = slot(api, .Session_Get_Output_Name, type_of(rt.output_name))
	rt.create_tensor = slot(api, .Create_Tensor_With_Data, type_of(rt.create_tensor))
	rt.tensor_data = slot(api, .Get_Tensor_Mutable_Data, type_of(rt.tensor_data))
	rt.element_type = slot(api, .Get_Tensor_Element_Type, type_of(rt.element_type))
	rt.dims_count = slot(api, .Get_Dimensions_Count, type_of(rt.dims_count))
	rt.dims = slot(api, .Get_Dimensions, type_of(rt.dims))
	rt.type_and_shape = slot(api, .Get_Tensor_Type_And_Shape, type_of(rt.type_and_shape))
	rt.create_cpu_info = slot(api, .Create_Cpu_Memory_Info, type_of(rt.create_cpu_info))
	rt.allocator_free = slot(api, .Allocator_Free, type_of(rt.allocator_free))
	rt.default_alloc = slot(api, .Get_Allocator_With_Default_Options, type_of(rt.default_alloc))
	rt.release_env = slot(api, .Release_Env, type_of(rt.release_env))
	rt.release_status = slot(api, .Release_Status, type_of(rt.release_status))
	rt.release_memory = slot(api, .Release_Memory_Info, type_of(rt.release_memory))
	rt.release_session = slot(api, .Release_Session, type_of(rt.release_session))
	rt.release_value = slot(api, .Release_Value, type_of(rt.release_value))
	rt.release_info = slot(api, .Release_Tensor_Type_And_Shape_Info, type_of(rt.release_info))
	rt.release_options = slot(api, .Release_Session_Options, type_of(rt.release_options))

	WARNING :: 2
	check(rt, rt.create_env(WARNING, "ottis", &rt.env), "creating environment") or_return
	DEVICE_ARENA :: 1
	CPU_DEFAULT :: 0
	check(rt, rt.create_cpu_info(DEVICE_ARENA, CPU_DEFAULT, &rt.memory), "creating memory info") or_return
	check(rt, rt.default_alloc(&rt.allocator), "getting allocator") or_return
	return nil
}

destroy :: proc(rt: ^Runtime) {
	if rt.memory != nil do rt.release_memory(rt.memory)
	if rt.env != nil do rt.release_env(rt.env)
	// The library stays loaded: ORT's own threads may outlive the env.
	rt^ = {}
}

Model :: struct {
	rt:      ^Runtime,
	session: ^Session,
	inputs:  []cstring, // names, owned
	outputs: []cstring,
}

Options :: struct {
	intra_threads: int,
}

load_model :: proc(rt: ^Runtime, path: string, options: Options) -> (m: Model, err: Error) {
	m.rt = rt
	so: ^Session_Options
	check(rt, rt.create_options(&so), "session options") or_return
	defer rt.release_options(so)
	SEQUENTIAL :: 0
	ENABLE_ALL :: 99
	check(rt, rt.set_exec_mode(so, SEQUENTIAL), "execution mode") or_return
	check(rt, rt.set_opt_level(so, ENABLE_ALL), "optimization level") or_return
	check(rt, rt.set_intra(so, c.int(max(options.intra_threads, 1))), "intra-op threads") or_return
	check(rt, rt.set_inter(so, 1), "inter-op threads") or_return

	// ORT takes paths as ORTCHAR_T: UTF-16 on Windows, UTF-8 elsewhere.
	when ODIN_OS == .Windows {
		wide := utf16_cstring(path)
		defer delete(wide)
		check(rt, rt.create_session(rt.env, raw_data(wide), so, &m.session), path) or_return
	} else {
		cpath := strings.clone_to_cstring(path, context.temp_allocator)
		check(rt, rt.create_session(rt.env, rawptr(cpath), so, &m.session), path) or_return
	}

	n: c.size_t
	check(rt, rt.input_count(m.session, &n), "input count") or_return
	m.inputs = make([]cstring, n)
	for i in 0 ..< n {
		name: cstring
		check(rt, rt.input_name(m.session, i, rt.allocator, &name), "input name") or_return
		m.inputs[i] = strings.clone_to_cstring(string(name))
		rt.allocator_free(rt.allocator, rawptr(name))
	}
	check(rt, rt.output_count(m.session, &n), "output count") or_return
	m.outputs = make([]cstring, n)
	for i in 0 ..< n {
		name: cstring
		check(rt, rt.output_name(m.session, i, rt.allocator, &name), "output name") or_return
		m.outputs[i] = strings.clone_to_cstring(string(name))
		rt.allocator_free(rt.allocator, rawptr(name))
	}
	return m, nil
}

unload_model :: proc(m: ^Model) {
	if m.session != nil do m.rt.release_session(m.session)
	for name in m.inputs do delete(name)
	for name in m.outputs do delete(name)
	delete(m.inputs)
	delete(m.outputs)
	m^ = {}
}

input_index :: proc(m: ^Model, name: string) -> int {
	for n, i in m.inputs do if string(n) == name do return i
	return -1
}

output_index :: proc(m: ^Model, name: string) -> int {
	for n, i in m.outputs do if string(n) == name do return i
	return -1
}

// Runs with one value per model input, in the model's input order, and
// fills one value per model output (ORT-allocated; the caller releases).
run :: proc(m: ^Model, inputs: []^Value, outputs: []^Value) -> Error {
	assert(len(inputs) == len(m.inputs) && len(outputs) == len(m.outputs))
	for &o in outputs do o = nil
	return check(
		m.rt,
		m.rt.run(m.session, nil, raw_data(m.inputs), raw_data(inputs), len(inputs), raw_data(m.outputs), len(outputs), raw_data(outputs)),
		"run",
	)
}

// A tensor over caller memory; the memory must outlive the value. Zero-size
// tensors still need a valid pointer.
tensor :: proc(rt: ^Runtime, data: rawptr, bytes: int, shape: []i64, type: Element_Type) -> (v: ^Value, err: Error) {
	@(static) empty: u64
	ptr := data if data != nil else rawptr(&empty)
	check(rt, rt.create_tensor(rt.memory, ptr, c.size_t(bytes), raw_data(shape), len(shape), type, &v), "tensor") or_return
	return v, nil
}

tensor_f32 :: proc(rt: ^Runtime, data: []f32, shape: []i64) -> (^Value, Error) {
	return tensor(rt, raw_data(data), len(data) * size_of(f32), shape, .Float32)
}

tensor_i64 :: proc(rt: ^Runtime, data: []i64, shape: []i64) -> (^Value, Error) {
	return tensor(rt, raw_data(data), len(data) * size_of(i64), shape, .Int64)
}

release :: proc(rt: ^Runtime, v: ^Value) {
	if v != nil do rt.release_value(v)
}

// The shape (into `buf`) and element data of an ORT tensor.
Tensor_View :: struct {
	type:  Element_Type,
	shape: []i64,
	data:  rawptr,
	count: int,
}

view :: proc(rt: ^Runtime, v: ^Value, buf: []i64) -> (t: Tensor_View, err: Error) {
	info: ^Tensor_Info
	check(rt, rt.type_and_shape(v, &info), "tensor info") or_return
	defer rt.release_info(info)
	check(rt, rt.element_type(info, &t.type), "element type") or_return
	rank: c.size_t
	check(rt, rt.dims_count(info, &rank), "rank") or_return
	if int(rank) > len(buf) do return t, fmt.aprintf("tensor rank %d exceeds %d", rank, len(buf))
	check(rt, rt.dims(info, raw_data(buf), rank), "dims") or_return
	t.shape = buf[:rank]
	t.count = 1
	for d in t.shape do t.count *= int(d)
	check(rt, rt.tensor_data(v, &t.data), "tensor data") or_return
	return t, nil
}

view_f32 :: proc(t: Tensor_View) -> []f32 {
	assert(t.type == .Float32)
	return ([^]f32)(t.data)[:t.count]
}

@(private)
utf16_cstring :: proc(s: string) -> []u16 {
	out := make([dynamic]u16, 0, len(s) + 1)
	for r in s {
		if r >= 0x10000 {
			v := u32(r) - 0x10000
			append(&out, u16(0xd800 + (v >> 10)), u16(0xdc00 + (v & 0x3ff)))
		} else {
			append(&out, u16(r))
		}
	}
	append(&out, 0)
	return out[:]
}

