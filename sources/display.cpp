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

#ifndef TERMSHADER_DISPLAY_CPP_

#include "display.hpp"

#include <assert.h>
#include <termios.h>
#include <unistd.h>

#include <codecvt>
#include <cstring>
#include <iostream>
#include <locale>
#include <string>

#include "maths.hpp"

#include "itou32.cpp"


using namespace ts;


static const char cClearU8[] = "\033[2J";


static Rcode write_all(int fd, const char* s, u32 sz, std::ostream& los) {

    while (sz > 0) {
        ssize_t nb = write(fd, s, sz);
        if (nb < 0) {
            APPEND_ERROR(los, "'write' failed with %s", strerror(errno));
            return R(IoError);
        }
        s += nb;
        sz -= nb;
    }

    return R(Ok);
}


Display::Display() noexcept
    : _chars(nullptr), _szchars(0)
    , _ixready(0)
{}


Display::~Display() noexcept {

    tearDown();
}


bool Display::valid() const noexcept {

    return _chars != nullptr;
}


Rcode Display::setUp(u32 width, u32 height) noexcept {

    if (valid()) return R(Already);

    _bufs[0].w = width;
    _bufs[0].h = height;
    _bufs[0].stride = width;
    _bufs[0].data = new cell_t[width * height];
    std::memset(_bufs[0].data, 0, sizeof(cell_t)*width*height);

    _bufs[1] = _bufs[0];
    _bufs[1].data = new cell_t[width * height];
    std::memset(_bufs[1].data, 0, sizeof(cell_t)*width*height);

    const std::u32string cX = U"\033[38;2;255;255;255m\033[48;2;255;255;255m\033[0m";

    _szchars = ((width+16)*height+1)*cX.size()+lengthof(cClearU8);
    _chars = new char32_t[_szchars];
    std::memset(_chars, 0, sizeof(char32_t)*_szchars);

    _ixready = 0;

    return R(Ok);
}


Rcode Display::tearDown() noexcept {

    if (!valid()) return R(Already);

    delete[] _bufs[0].data;
    delete[] _bufs[1].data;

    delete[] _chars;
    _chars = nullptr;
    _szchars = 0;

    return R(Ok);
}


color_buffer_f& Display::buffer() noexcept {

    if (!valid()) {
        static color_buffer_f scb{0};
        return scb;
    }

    return _bufs[(_ixready+1)%2];
}


Rcode Display::flip() noexcept {

    std::scoped_lock<std::mutex> sl(_ixreadyM);
    _ixready = (_ixready+1)%2;

    return R(Ok);
}


static u32 fill_color(char32_t* buffer, u32 pos, const char32_t* layer, const v4f& c) {

    buffer[pos++] = U'\033';
    buffer[pos++] = U'[';
    buffer[pos++] = layer[0];
    buffer[pos++] = layer[1];
    buffer[pos++] = U';';
    buffer[pos++] = U'2';
    buffer[pos++] = U';';

    v3i color = floor(c.xyz * 255.0f);
    clamp(color, 0, 255);

    const std::u32string& sr = cByteToU32[color.r];
    const std::u32string& sg = cByteToU32[color.g];
    const std::u32string& sb = cByteToU32[color.b];

    std::memcpy(buffer+pos, sr.c_str(), sr.size()*sizeof(char32_t));
    pos += sr.size();
    buffer[pos++] = U';';

    std::memcpy(buffer+pos, sg.c_str(), sg.size()*sizeof(char32_t));
    pos += sg.size();
    buffer[pos++] = U';';
    std::memcpy(buffer+pos, sb.c_str(), sb.size()*sizeof(char32_t));
    pos += sb.size();
    buffer[pos++] = U'm';

    return pos;
}


Rcode Display::output(FILE* f, std::ostream& los) noexcept {

    if (!valid()) return R(Uninit);

    Stopwatch<float> sw;
    sw.reset();
    {
        std::scoped_lock<decltype(_ixreadyM)> sl(_ixreadyM);

        color_buffer_f& b = _bufs[_ixready];
        u32 maxY = b.h * b.stride;
        u32 i = 0;

        static const char32_t cReturnToStart[] = U"\033[1;0f";
        static_assert(sizeof(cReturnToStart) == 7*sizeof(char32_t));

        std::memcpy(_chars+i, cReturnToStart, sizeof(cReturnToStart)-sizeof(*cReturnToStart));
        i += 6;
        v3f lastFg, lastBg;
        for (u32 y = 0; y < maxY; y += b.stride) {
            assert(i < _szchars);
            u32 maxX = y+b.w;
            for (u32 x = y; x < maxX; ++x) {
                cell_t& c = b.data[x];
                if (i != 0) {
                    if (lastFg != c.fg.xyz) {
                        i = fill_color(_chars, i, U"38", c.fg);
                        lastFg = c.fg.xyz;
                    }
                    if (lastBg != c.bg.xyz) {
                        i = fill_color(_chars, i, U"48", c.bg);
                        lastBg = c.bg.xyz;
                    }
                } else {
                    i = fill_color(_chars, i, U"38", c.fg);
                    i = fill_color(_chars, i, U"48", c.bg);
                    lastFg = c.fg.xyz;
                    lastBg = c.bg.xyz;
                }
                _chars[i++] = static_cast<char32_t>(c.c);
            }
            _chars[i++] = U'\n';
        }
        _chars[i++] = U'\033';
        _chars[i++] = U'[';
        _chars[i++] = U'0';
        _chars[i++] = U'm';
        _chars[i++] = U'\0';
    }
    f32 elapsed1 = sw.measure();
    sw.reset();
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    std::string bs = conv.to_bytes(_chars);
    f32 elapsed2 = sw.measure();
    sw.reset();

    u32 sz = bs.size();
    char* ob = (char*)bs.data();

    fflush(f);

    FWDR(write_all, fileno(f), ob, sz, los);
    f32 elapsed3 = sw.measure();

    if (0 != fflush(f)) {
        APPEND_ERROR(los, "FFLUSH failed!");
        return R(LogicError);
    }
    tcdrain(fileno(f));

    std::cout << "SZ: " << sz << "; T1: " << elapsed1*1000.0f << "; T2: " << elapsed2*1000.0f << "; T3: " << elapsed3*1000.0f << std::endl;

    return R(Ok);
}




Rcode Display::clear(FILE* f, std::ostream& los) noexcept {

    if (!valid()) return R(Uninit);

    FWDR(write_all, fileno(f), cClearU8, lengthof(cClearU8), los);
    if (0 != fflush(f)) {
        APPEND_ERROR(los, "'fflush' failed!");
        return R(IoError);
    }

    return R(Ok);
}


#define TERMSHADER_DISPLAY_CPP_
#endif

