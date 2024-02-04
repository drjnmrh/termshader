/**
 * MIT License
 *
 * Copyright (c) 2024 Stoned Fox
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef TERMSHADER_GLOBAL_HPP_


#include <stddef.h>
#include <stdint.h>

#include <chrono>
#include <cstdio>


namespace ts {;


enum class Rcode {
    Ok           = 0
,   CudaError    = 1
,   CudaSetup    = 2
,   Timeout      = 3
,   MemError     = 4
,   InvalidInput = 5
,   LogicError   = 6
,   Already      = 7
,   Uninit       = 8
,   IoError      = 9
,   Unknown
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


template < typename T, size_t N >
static constexpr size_t lengthof(T (&a)[N]) { return N; }

#define KB(Nb) (1024*(Nb))
#define MB(Nb) (1024*KB(Nb))

#define APPEND_INFO(Los, Msg, ...) \
    do {\
        char buffer[256];\
        std::snprintf(buffer, ts::lengthof(buffer), Msg, ##__VA_ARGS__);\
        (Los) << "\033[34mINFO  \033[0m: " << buffer << std::endl;\
    }while(0)

#define APPEND_ERROR(Los, Msg, ...) \
    do {\
        char buffer[256];\
        std::snprintf(buffer, ts::lengthof(buffer), Msg, ##__VA_ARGS__);\
        (Los) << "\033[31mFAILED\033[0m: " << buffer << std::endl;\
    }while(0);


using byte = uint8_t;
using u16  = uint16_t;
using i16  = int16_t;
using u32  = uint32_t;
using i32  = int32_t;
using u64  = uint64_t;
using i64  = int64_t;
using f32  = float;
using f64  = double;


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
