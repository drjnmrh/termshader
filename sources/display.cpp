#ifndef TERMSHADER_DISPLAY_CPP_

#include "display.hpp"

#include <assert.h>

#include <codecvt>
#include <cstring>
#include <iostream>
#include <string>

#include "maths.hpp"

#include "itou32.cpp"


using namespace ts;


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
	const std::u32string cClear = U"\033[2J\033[0;0H";
	_szchars = ((width+16)*height+1)*cX.size()+cClear.size();
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


Rcode Display::output(std::ostream& ostr) noexcept {

	if (!valid()) return R(Uninit);

	Stopwatch<float> sw;
	sw.reset();

	std::scoped_lock<decltype(_ixreadyM)> sl(_ixreadyM);

	color_buffer_f& b = _bufs[_ixready];
	u32 maxY = b.h * b.stride;
	u32 i = 0;
	_chars[i++] = U'\033';
	_chars[i++] = U'[';
	_chars[i++] = U'2';
	_chars[i++] = U'J';
	std::memcpy(_chars+i, U"\033[1;0f", 6*sizeof(char32_t));
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

	std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
	std::string bs = conv.to_bytes(_chars);

	float t1 = sw.measure();

	sw.reset();
	//ostr.flush();
	ostr.write(bs.c_str(), bs.size());
	ostr.flush();
	float t2 = sw.measure();

	//ostr << "T1: " << t1 << "; T2: " << t2 << std::endl;

	return R(Ok);
}

#define TERMSHADER_DISPLAY_CPP_
#endif

