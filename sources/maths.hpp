#ifndef TERMSHADER_MATHS_HPP_


#include <cmath>

#include "global.hpp"


namespace ts {;


template < u32 D, typename T >
struct vec;


template < typename T >
struct vec<2, T> {
	union {
		struct { T x, y; };
		T e[2];
	};
};
using v2f = vec<2, f32>;
using v2i = vec<2, i32>;

template < typename T >
struct vec<3, T> {
	union {
		struct { T x, y, z; };
		struct { T r, g, b; };
		struct { vec<2, T> xy; T _; };
		struct { T _0; vec<2, T> yz; };
		T e[3];
	};
};
using v3f = vec<3, f32>;
using v3i = vec<3, i32>;


template < typename T >
struct vec<4, T> {
	union {
		struct { T r, g, b, a; };
		struct { T x, y, z, w; };
		struct { vec<2, T> xy, zw; };
		struct { vec<3, T> xyz; T _; };
		T e[4];
	};
};
using v4f = vec<4, f32>;


template < typename T > vec<2, T> vec2(T x, T y) noexcept {
	vec<2, T> res;
	res.x = x;
	res.y = y;
	return res;
}


template <u32 D, typename T> bool operator == (const vec<D, T>& a, const vec<D, T>& b) {
	for(u32 i=0; i<D; ++i) if (a.e[i] != b.e[i]) return false;
	return true;
}

template <u32 D, typename T> bool operator != (const vec<D, T>& a, const vec<D, T>& b) {
	return !operator == (a, b);
}


#define DEFINE_FI_OP(Op) \
	template <u32 D> static vec<D, f32> operator Op (const vec<D, f32>& a, const vec<D, i32>& b) {\
		vec<D, f32> res;\
		for (u32 i = 0; i < D; ++i) res.e[i] = a.e[i] Op static_cast<f32>(b.e[i]);\
		return res;\
	}\
	template <u32 D> static vec<D, f32> operator Op (const vec<D, i32>& a, const vec<D, f32>& b) {\
		vec<D, f32> res;\
		for (u32 i = 0; i < D; ++i) res.e[i] = static_cast<f32>(a.e[i]) Op b.e[i];\
		return res;\
	}

DEFINE_FI_OP(+);
DEFINE_FI_OP(-);
DEFINE_FI_OP(*);
DEFINE_FI_OP(/);


template <u32 D> static vec<D, i32> floor(const vec<D, f32>& a) noexcept {

	vec<D, i32> res;
	for (u32 i = 0; i < D; ++i) res.e[i] = static_cast<i32>(std::floor(a.e[i]));
	return res;
}


template <u32 D> static vec<D, i32> ceil(const vec<D, f32>& a) noexcept {

	vec<D, i32> res;
	for(u32 i=0; i<D; ++i) res.e[i] = static_cast<i32>(std::ceil(a.e[i]));
	return res;
}


template <u32 D, typename T> static void clamp(vec<D, T>& a, T m, T M) noexcept {

	for (u32 i = 0; i < D; ++i) {
		if (a.e[i] > M) a.e[i] = M;
		else if (a.e[i] < m) a.e[i] = m;
	}
}


template <u32 D, typename T> static vec<D, i32> operator % (const vec<D, i32>& a, T s) noexcept {

	vec<D, i32> res;
	for (u32 i = 0; i < D; ++i) res.e[i] = a.e[i] % s;
	return res;
}


#define DEFINE_VV_OP(Op) \
	template < u32 D, typename T > static vec<D, T> operator Op (const vec<D, T>& a, const vec<D, T>& b) {\
		vec<D, T> res;\
		for (u32 i = 0; i < D; ++i) res.e[i] = a.e[i] Op b.e[i];\
		return res;\
	}

DEFINE_VV_OP(+);
DEFINE_VV_OP(-);
DEFINE_VV_OP(*);
DEFINE_VV_OP(/);

#define DEFINE_VS_OP(Op) \
	template < u32 D, typename T > static vec<D, T> operator Op (const vec<D, T>& a, T s) {\
		vec<D, T> res;\
		for (u32 i = 0; i < D; ++i) res.e[i] = a.e[i] Op s;\
		return res;\
	}

DEFINE_VS_OP(+);
DEFINE_VS_OP(-);
DEFINE_VS_OP(*);
DEFINE_VS_OP(/);

#define DEFINE_SV_OP(Op) \
	template <u32 D, typename T> static vec<D, T> operator Op (T s, const vec<D, T>& a) {\
		vec<D, T> res;\
		for (u32 i = 0; i < D; ++i) res.e[i] = s Op a.e[i];\
		return res;\
	}

DEFINE_SV_OP(*);

template <u32 D, typename T> static vec<D, T> operator - (const vec<D, T>& a) {
	vec<D, T> res;
	for (u32 i = 0; i < D; ++i) res.e[i] = -a.e[i];
	return res;
}


template <u32 D> static vec<D, f32> operator / (const vec<D, f32>& a, i32 s) noexcept {
	vec<D, f32> res;
	for (u32 i=0; i<D; ++i) res.e[i] = a.e[i]/(f32)s;
	return res;
}


}


#define TERMSHADER_MATHS_HPP_
#endif
