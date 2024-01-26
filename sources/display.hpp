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

#ifndef TERMSHADER_DISPLAY_HPP_


#include <iostream>
#include <mutex>

#include "global.hpp"
#include "maths.hpp"


namespace ts {;


struct cell_t {
    v4f fg;
    v4f bg;
    u32 c;
};


struct color_buffer_f {
    u32 w, h, stride;
    cell_t* data;
};


enum class BufferType : u32 {
    Screen = 0
,   Backbuffer = 1
};


class Display {
public:
    Display() noexcept;
   ~Display() noexcept;

    bool valid() const noexcept;

    Rcode setUp(u32 width, u32 height) noexcept;
    Rcode tearDown() noexcept;

    color_buffer_f& buffer() noexcept;

    Rcode flip() noexcept;

    Rcode output(FILE* f, std::ostream& los) noexcept;
    Rcode clear(FILE* f, std::ostream& los) noexcept;

private:
    color_buffer_f _bufs[2];

    char32_t* _chars;
    u32 _szchars;

    u32 _ixready;
    std::mutex _ixreadyM;
};


}


#define TERMSHADER_DISPLAY_HPP_
#endif
