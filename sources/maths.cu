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

#ifndef TERMSHADER_MATHS_CU_


#include "maths.hpp"


using namespace ts;


__device__ static vec<2, f32> vec2_d(f32 x, f32 y) {
	vec<2, f32> res;
	res.x = x;
	res.y = y;
	return res;
}

__device__ static v2f vec2_d(f32 s) { return vec2_d(s, s); }

__device__ static v3f vec3_d(f32 x, f32 y, f32 z) {

	v3f res;
	res.x = x;
	res.y = y;
	res.z = z;
	return res;
}

__device__ static v3f vec3_d(const v2f& v, f32 z) {
	return vec3_d(v.x, v.y, z);
}

__device__ static v3f vec3_d(f32 s) {
	return vec3_d(s, s, s);
}

__device__ static v4f vec4_d(f32 x, f32 y, f32 z, f32 w) {
	v4f r; r.x=x; r.y=y; r.z=z; r.w=w;
	return r;
}

__device__ static v4f vec4_d(const v3f& v, f32 w) { return vec4_d(v.x, v.y, v.z, w); }

__device__ static v2f floor_d(const v2f& a) {
	v2f r;
	r.x = floorf(a.x);
	r.y = floorf(a.y);
	return r;
}

__device__ static v3f floor_d(const v3f& a) {

	v3f res;
	res.x = floor(a.x);
	res.y = floor(a.y);
	res.z = floor(a.z);
	return res;
}

__device__ static v4f floor_d(const v4f& a) {
	v4f r;
	r.x = floor(a.x);
	r.y = floor(a.y);
	r.z = floor(a.z);
	r.w = floor(a.w);
	return r;
}

__device__ static f32 dot_d(const v2f& a, const v2f& b) {
	return a.x*b.x+a.y*b.y;
}

__device__ static f32 dot_d(const v3f& a, const v3f& b) {
	return a.x*b.x+a.y*b.y+a.z*b.z;
}

__device__ static f32 dot_d(const v4f& a, const v4f& b) {
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__device__ static v2f operator * (const v2f& a, const v2f& b) {
	v2f r; r.x=a.x*b.x; r.y=a.x*b.y;
	return r;
}

__device__ static v3f operator * (const v3f& a, f32 s) {
	v3f res; res.x=a.x*s; res.y=a.y*s; res.z=a.z*s;
	return res;
}

__device__ static v4f operator * (const v4f& a, f32 s) {
	v4f r; r.xyz=a.xyz*s;r.w=a.w*s;
	return r;
}

__device__ static v4f operator * (const v4f& a, const v4f& b) {
	v4f r; r.x=a.x*b.x; r.y=a.y*b.y; r.z=a.z*b.z; r.w=a.w*b.w;
	return r;
}

__device__ static v3f operator * (const v3f& a, const v3f& b) {
	v3f r; r.x=a.x*b.x; r.y=a.y*b.y; r.z=a.z*b.z;
	return r;
}

__device__ static v4f operator * (f32 s, const v4f& b) {
	v4f r; r.x=s*b.x; r.y=s*b.y; r.z=s*b.z; r.w=s*b.w;
	return r;
}

__device__ static v3f operator * (f32 s, const v3f& b) {
	v3f r; r.x=s*b.x; r.y=s*b.y; r.z=s*b.z;
	return r;
}

__device__ static v2f operator * (f32 s, const v2f& b) {
	v2f r; r.x=s*b.x; r.y=s*b.y;
	return r;
}

__device__ static v2f operator * (const v2f& a, f32 s) {
	v2f r; r.x=a.x*s; r.y=a.y*s;
	return r;
}

__device__ static v2f operator + (const v2f& a, const v2f& b) {
	v2f r; r.x=a.x+b.x; r.y=a.y+b.y;
	return r;
}

__device__ static v4f operator + (const v4f& a, f32 s) {
	v4f r; r.x=a.x+s; r.y=a.y+s; r.z=a.z+s; r.w=a.w+s;
	return r;
}

__device__ static v4f operator + (f32 s, const v4f& b) {
	v4f r; r.x=s+b.x; r.y=s+b.y; r.z=s+b.z; r.w=s+b.w;
	return r;
}

__device__ static v4f operator + (const v4f&a, const v4f& b) {
	v4f r; r.x=a.x+b.x; r.y=a.y+b.y; r.z=a.z+b.z; r.w=a.w+b.w;
	return r;
}

__device__ static v3f operator + (const v3f& a, f32 s) {
	v3f r; r.x=a.x+s; r.y=a.y+s; r.z=a.z+s;
	return r;
}

__device__ static v3f operator + (f32 s, const v3f& a) {
	v3f r; r.x=s+a.x; r.y=s+a.y; r.z=s+a.z;
	return r;
}

__device__ static v3f operator + (const v3f& a, const v3f& b) {
	v3f r; r.x=a.x+b.x; r.y=a.y+b.y; r.z=a.z+b.z;
	return r;
}

__device__ static v4f operator - (const v4f& a) {
	v4f r; r.x=-a.x; r.y=-a.y; r.z=-a.z; r.w=-a.w;
	return r;
}

__device__ static v2f operator - (const v2f& a, const v2f& b) {
	v2f r; r.x=a.x-b.x; r.y = a.y-b.y;
	return r;
}

__device__ static v3f operator - (const v3f& a, const v3f& b) {
	v3f r; r.x=a.x-b.x;r.y=a.y-b.y;r.z=a.z-b.z;
	return r;
}

__device__ static v4f operator - (const v4f& a, const v4f& b) {
	v4f r; r.x=a.x-b.x; r.y=a.y-b.y; r.z=a.z-b.z; r.w=a.w-b.w;
	return r;
}

__device__ static v4f operator - (const v4f& a, f32 s) {
	v4f r; r.x=a.x-s; r.y=a.y-s; r.z=a.z-s; r.w=a.w-s;
	return r;
}

__device__ static v4f operator - (f32 s, const v4f& b) {
	v4f r; r.x=s-b.x; r.y=s-b.y; r.z=s-b.z; r.w=s-b.w;
	return r;
}

__device__ static v3f operator - (f32 s, const v3f& b) {
	v3f r; r.x=s-b.x; r.y=s-b.y; r.z=s-b.z;
	return r;
}

__device__ static v2f operator / (const v2f& a, f32 s) {
	return vec2_d(a.x/s, a.y/s);
}

__device__ static v3f operator / (const v3f& a, f32 s) {
	v3f r; r.x=a.x/s;r.y=a.y/s;r.z=a.z/s;
	return r;
}

__device__ static v4f operator / (const v4f& a, f32 s) {
	v4f r; r.x=a.x/s; r.y=a.y/s; r.z=a.z/s; r.w=a.w/s;
	return r;
}

__device__ static v4f& operator *= (v4f& a, const v4f& b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

__device__ static v3f& operator *= (v3f& a, f32 s) {
	a.x *= s;
	a.y *= s;
	a.z *= s;
	return a;
}

__device__ static v2f& operator += (v2f& a, f32 s) {
	a.x += s;
	a.y += s;
	return a;
}

__device__ static v2f& operator += (v2f& a, const v2f& b) {
	a.x += b.x;
	a.y += b.y;
	return a;
}

__device__ static v2f& operator -= (v2f& a, const v2f& b) {
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

__device__ static v3f step_d(const v3f& edge, const v3f& x) {
	v3f r;
	r.x = x.x < edge.x ? 0.0f : 1.0f;
	r.y = x.y < edge.y ? 0.0f : 1.0f;
	r.z = x.z < edge.z ? 0.0f : 1.0f;
	return r;
}

__device__ static v4f step_d(const v4f& edge, const v4f& x) {
	v4f r;
	r.x = x.x < edge.x ? 0.0f : 1.0f;
	r.y = x.y < edge.y ? 0.0f : 1.0f;
	r.z = x.z < edge.z ? 0.0f : 1.0f;
	r.w = x.w < edge.w ? 0.0f : 1.0f;
	return r;
}

__device__ static v3f min_d(const v3f& a, const v3f& b) {
	v3f r;
	r.x = a.x < b.x ? a.x : b.x;
	r.y = a.y < b.y ? a.y : b.y;
	r.z = a.z < b.z ? a.z : b.z;
	return r;
}

__device__ static v4f min_d(const v4f& a, const v4f& b) {
	v4f r;
	r.x = min(a.x, b.x);
	r.y = min(a.y, b.y);
	r.z = min(a.z, b.z);
	r.w = min(a.w, b.w);
	return r;
}

__device__ static v3f max_d(const v3f& a, const v3f& b) {
	v3f r;
	r.x = a.x > b.x ? a.x : b.x;
	r.y = a.y > b.y ? a.y : b.y;
	r.z = a.z > b.z ? a.z : b.z;
	return r;
}

__device__ static v4f max_d(const v4f& a, const v4f& b) {
	v4f r;
	r.x = max(a.x, b.x);
	r.y = max(a.y, b.y);
	r.z = max(a.z, b.z);
	r.w = max(a.w, b.w);
	return r;
}

__device__ static v4f abs_d(const v4f& a) {
	v4f r;
	r.x = a.x > 0.0f ? a.x : -a.x;
	r.y = a.y > 0.0f ? a.y : -a.y;
	r.z = a.z > 0.0f ? a.z : -a.z;
	r.w = a.w > 0.0f ? a.w : -a.w;
	return r;
}

__device__ static v4f rsqrt_d(const v4f& a) {
	v4f r;
	r.x = rsqrtf(a.x);
	r.y = rsqrtf(a.y);
	r.z = rsqrtf(a.z);
	r.w = rsqrtf(a.w);
	return r;
}

__device__ static v2f fract_d(const v2f& a) {
	return a - floor_d(a);
}

__device__ static f32 fract_d(f32 a) {
	return a - floorf(a);
}

__device__ static f32 clamp_d(f32 x, f32 min, f32 max) {

	if (min > max) { f32 tmp = min; min = max; max = tmp; }
	if (x < min) x = min;
	else if (x > max) x = max;
	return x;
}

__device__ static v2f mod_d(const v2f& x, f32 y) {
	return x - y*floor_d(x/y);
}

__device__ static f32 mod_d(f32 x, f32 y) {
	return x - y*floor(x/y);
}

__device__ static f32 length_d(const v2f& x) { return sqrt(x.x*x.x+x.y*x.y); }


#define TERMSHADER_MATHS_CU_
#endif
