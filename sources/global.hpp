#ifndef TERMSHADER_GLOBAL_HPP_


#include <stddef.h>
#include <stdint.h>

#include <chrono>


namespace ts {;


enum class Rcode {
	Ok           = 0
,	CudaError    = 1
,	CudaSetup    = 2
,	Timeout      = 3
,	MemError     = 4
,	InvalidInput = 5
,	LogicError   = 6
,	Already      = 7
,	Uninit       = 8
,	Unknown
};
#define R(Rc) ts::Rcode:: Rc
#define R2I(Rc) static_cast<int>(ts::Rcode:: Rc);
#define FAILED(Rc) ((Rc) != R(Ok))

#define FWDI(Func, ...) \
	do {\
		ts::Rcode rc = Func(__VA_ARGS__);\
		if (FAILED(rc)) {\
			return static_cast<int>(rc);\
		}\
	}while(0)

#define FWDR(Func, ...) \
	do {\
		ts::Rcode rc = Func(__VA_ARGS__);\
		if (FAILED(rc)) {\
			return rc;\
		}\
	}while(0)


using byte = uint8_t;
using u16  = uint16_t;
using i16  = int16_t;
using u32  = uint32_t;
using i32  = int32_t;
using u64  = uint64_t;
using i64  = int64_t;
using f32  = float;
using f64  = double;


template < typename T, size_t N >
static constexpr size_t lengthof(T (&a)[N]) { return N; }


template < typename T >
class Stopwatch {
	using defclock = std::chrono::high_resolution_clock;
public:
	void reset() noexcept {

		_tps = defclock::now();
	}

	T measure() noexcept {

		defclock::time_point tpn = defclock::now();
		std::chrono::duration<T> d = tpn - _tps;
		return d.count();
	}
private:
	defclock::time_point _tps;
};


}


#define TERMSHADER_GLOBAL_HPP_
#endif
