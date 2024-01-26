#ifndef TERMSHADER_DISPLAY_HPP_


#include <atomic>
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
,	Backbuffer = 1
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
