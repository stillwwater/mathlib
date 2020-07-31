// Copyright (c) 2020 stillwwater
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#ifndef MATHF_H
#define MATHF_H

#include <type_traits>
#include <algorithm> // min, max
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdio>

#include "config.h"

#ifdef MATHF_SSE
#include <xmmintrin.h>
#endif

#define PI 3.14159265358979
#define E  2.71828182845904

namespace mathf {

constexpr float Epsilon  = 1.0e-6f;
constexpr float DegToRad = PI / 180.0;
constexpr float RadToDeg = 180.0 / PI;

// Makes certain code easier to read.
constexpr int AxisX = 0;
constexpr int AxisY = 1;
constexpr int AxisZ = 2;
constexpr int AxisW = 3;

namespace internal {
template <typename T>
struct Vector2_t;

template <typename T>
struct Vector3_t;

template <typename T>
struct Vector4_t;
} // internal

// 2D float vector
using float2 = internal::Vector2_t<float>;

// 3D float vector
using float3 = internal::Vector3_t<float>;

// 4D float vector
using float4 = internal::Vector4_t<float>;

// 2D int vector
using int2 = internal::Vector2_t<int32_t>;

// 3D int vector
using int3 = internal::Vector3_t<int32_t>;

// 4D int vector
using int4 = internal::Vector4_t<int32_t>;

namespace internal {

template <typename T>
struct Vector2_t {
    static_assert(std::is_arithmetic<T>(), "T must be a numeric type");
    T x, y;

    Vector2_t() = default;
    constexpr Vector2_t(T x, T y) : x{x}, y{y} {}

    template <typename U>
    explicit Vector2_t(const Vector2_t<U> &vec2)
        : x{static_cast<T>(vec2.x)}
        , y{static_cast<T>(vec2.y)} {}

    // Initializes all elements of the vector to s.
    explicit constexpr Vector2_t(T s) : x{s}, y{s} {}

    explicit Vector2_t(const Vector3_t<T> &vec3);
    explicit Vector2_t(const Vector4_t<T> &vec4);

    T operator[](int i) const;
    T &operator[](int i);

    bool operator==(const Vector2_t<T> &v) const;
    bool operator!=(const Vector2_t<T> &v) const;

    // All operations are component wise
    Vector2_t<T> operator-() const;
    Vector2_t<T> &operator+=(const Vector2_t<T> &v);
    Vector2_t<T> &operator-=(const Vector2_t<T> &v);
    Vector2_t<T> &operator*=(const Vector2_t<T> &v);
    Vector2_t<T> &operator/=(const Vector2_t<T> &v);
    Vector2_t<T> &operator*=(T s);
    Vector2_t<T> &operator/=(T s);

    // Returns the vector's magnitude
    float length() const;

    // Returns the vector's magnitude squared
    float length_sqr() const;
};

template <typename T>
struct Vector3_t {
    static_assert(std::is_arithmetic<T>(), "T must be a numeric type");
    T x, y, z;

    Vector3_t() = default;
    constexpr Vector3_t(T x, T y, T z) : x{x}, y{y}, z{z} {}

    template <typename U>
    explicit Vector3_t(const Vector3_t<U> &vec3)
        : x{static_cast<T>(vec3.x)}
        , y{static_cast<T>(vec3.y)}
        , z{static_cast<T>(vec3.z)} {}

    // Initializes all elements of the vector to s.
    explicit constexpr Vector3_t(T s) : x{s}, y{s}, z{s} {}

    Vector3_t(const Vector2_t<T> &vec2);
    explicit Vector3_t(const Vector4_t<T> &vec4);

    T operator[](int i) const;
    T &operator[](int i);

    bool operator==(const Vector3_t<T> &v) const;
    bool operator!=(const Vector3_t<T> &v) const;

    // All operations are component wise
    Vector3_t<T> operator-() const;
    Vector3_t<T> &operator+=(const Vector3_t<T> &v);
    Vector3_t<T> &operator-=(const Vector3_t<T> &v);
    Vector3_t<T> &operator*=(const Vector3_t<T> &v);
    Vector3_t<T> &operator/=(const Vector3_t<T> &v);
    Vector3_t<T> &operator*=(T s);
    Vector3_t<T> &operator/=(T s);

    // Returns the vector's magnitude
    float length() const;

    // Returns the vector's magnitude squared
    float length_sqr() const;
};

template <typename T>
struct alignas(16) Vector4_t {
    static_assert(std::is_arithmetic<T>(), "T must be a numeric type");
#ifdef MATHF_SSE
    static_assert((sizeof(T) * 4) == sizeof(__m128),
                  "sizeof(Vector4) must be 16 bytes");
    union {
        // Use with float vector
        __m128 m128;
        // Use with int vector
        __m128i m128i;
        struct { T x, y, z, w; };
    };

    Vector4_t<float>(__m128 m) : m128{m} {}
    Vector4_t<int>(__m128i m) : m128i{m} {}
#else
    T x, y, z, w;
#endif

    Vector4_t() = default;
    constexpr Vector4_t(T x, T y, T z, T w)
        : x{x}, y{y}, z{z}, w{w} {}

    template <typename U>
    explicit Vector4_t(const Vector4_t<U> &vec4)
        : x{static_cast<T>(vec4.x)}
        , y{static_cast<T>(vec4.y)}
        , z{static_cast<T>(vec4.z)}
        , w{static_cast<T>(vec4.w)} {}

    // Initializes all elements of the vector to s.
    explicit constexpr Vector4_t(T s) : x{s}, y{s}, z{s}, w{s} {}

    Vector4_t(const Vector2_t<T> &vec2);
    Vector4_t(const Vector3_t<T> &vec3);

    T operator[](int i) const;
    T &operator[](int i);

    bool operator==(const Vector4_t<T> &v) const;
    bool operator!=(const Vector4_t<T> &v) const;

    // All operations are component wise
    Vector4_t<T> operator-() const;
    Vector4_t<T> &operator+=(const Vector4_t<T> &v);
    Vector4_t<T> &operator-=(const Vector4_t<T> &v);
    Vector4_t<T> &operator*=(const Vector4_t<T> &v);
    Vector4_t<T> &operator/=(const Vector4_t<T> &v);
    Vector4_t<T> &operator*=(T s);
    Vector4_t<T> &operator/=(T s);

    // Returns the vector's magnitude
    float length() const;

    // Returns the vector's magnitude squared
    float length_sqr() const;
};

//
// Vector2
//

template <typename T>
inline Vector2_t<T> operator+(const Vector2_t<T> &a, const Vector2_t<T> &b) {
    return {a.x + b.x, a.y + b.y};
}

template <typename T>
inline Vector2_t<T> operator-(const Vector2_t<T> &a, const Vector2_t<T> &b) {
    return {a.x - b.x, a.y - b.y};
}

template <typename T>
inline Vector2_t<T> operator*(const Vector2_t<T> &a, const Vector2_t<T> &b) {
    return {a.x * b.x, a.y * b.y};
}

template <typename T>
inline Vector2_t<T> operator/(const Vector2_t<T> &a, const Vector2_t<T> &b) {
    return {a.x / b.x, a.y / b.y};
}

template <typename T>
inline Vector2_t<T> operator*(const Vector2_t<T> &v, float s) {
    return {v.x * s, v.y * s};
}

template <typename T>
inline Vector2_t<T> operator*(float s, const Vector2_t<T> &v) {
    return {s * v.x, s * v.y};
}

template <typename T>
inline Vector2_t<T> operator/(const Vector2_t<T> &v, float s) {
    return {v.x / s, v.y / s};
}

inline float2 operator/(const float2 &v, float s) {
    float invs = 1.0f / s;
    return {v.x * invs, v.y * invs};
}

template <>
inline Vector2_t<float>::Vector2_t(const Vector3_t<float> &vec3)
    : x{vec3.x}, y{vec3.y} {}

template <>
inline Vector2_t<float>::Vector2_t(const Vector4_t<float> &vec4)
    : x{vec4.x}, y{vec4.y} {}

template <>
inline Vector2_t<int>::Vector2_t(const Vector3_t<int> &vec3)
    : x{vec3.x}, y{vec3.y} {}

template <>
inline Vector2_t<int>::Vector2_t(const Vector4_t<int> &vec4)
    : x{vec4.x}, y{vec4.y} {}

template <typename T>
inline T Vector2_t<T>::operator[](int i) const {
    assert(i >= 0 && i < 2);
    return (reinterpret_cast<const T *>(this))[i];
}

template <typename T>
inline T &Vector2_t<T>::operator[](int i) {
    assert(i >= 0 && i < 2);
    return (reinterpret_cast<T *>(this))[i];
}

template <typename T>
inline bool Vector2_t<T>::operator==(const Vector2_t<T> &v) const {
    return x == v.x && y == v.y;
}

template <typename T>
inline bool Vector2_t<T>::operator!=(const Vector2_t<T> &v) const {
    return !(*this == v);
}

template <typename T>
inline Vector2_t<T> Vector2_t<T>::operator-() const {
    return {-x, -y};
}

template <typename T>
inline Vector2_t<T> &Vector2_t<T>::operator+=(const Vector2_t<T> &v) {
    x += v.x;
    y += v.y;
    return *this;
}

template <typename T>
inline Vector2_t<T> &Vector2_t<T>::operator-=(const Vector2_t<T> &v) {
    x -= v.x;
    y -= v.y;
    return *this;
}

template <typename T>
inline Vector2_t<T> &Vector2_t<T>::operator*=(const Vector2_t<T> &v) {
    x *= v.x;
    y *= v.y;
    return *this;
}

template <typename T>
inline Vector2_t<T> &Vector2_t<T>::operator/=(const Vector2_t<T> &v) {
    x /= v.x;
    y /= v.y;
    return *this;
}

template <typename T>
inline Vector2_t<T> &Vector2_t<T>::operator*=(T s) {
    x *= s;
    y *= s;
    return *this;
}

template <typename T>
inline Vector2_t<T> &Vector2_t<T>::operator/=(T s) {
    x /= s;
    y /= s;
    return *this;
}

template <typename T>
inline float Vector2_t<T>::length() const {
    return sqrtf(x * x + y * y);
}

template <typename T>
inline float Vector2_t<T>::length_sqr() const {
    return x * x + y * y;
}

//
// Vector3
//

template <typename T>
inline Vector3_t<T> operator+(const Vector3_t<T> &a, const Vector3_t<T> &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <typename T>
inline Vector3_t<T> operator-(const Vector3_t<T> &a, const Vector3_t<T> &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <typename T>
inline Vector3_t<T> operator*(const Vector3_t<T> &a, const Vector3_t<T> &b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

template <typename T>
inline Vector3_t<T> operator/(const Vector3_t<T> &a, const Vector3_t<T> &b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

template <typename T>
inline Vector3_t<T> operator*(const Vector3_t<T> &v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

template <typename T>
inline Vector3_t<T> operator*(float s, const Vector3_t<T> &v) {
    return {s * v.x, s * v.y, s * v.z};
}

template <typename T>
inline Vector3_t<T> operator/(const Vector3_t<T> &v, float s) {
    return {v.x / s, v.y / s, v.z / s};
}

template <>
inline float3 operator/(const float3 &v, float s) {
    float invs = 1.0f / s;
    return {v.x * invs, v.y * invs, v.z * invs};
}

template <>
inline Vector3_t<float>::Vector3_t(const Vector4_t<float> &vec4)
    : x{vec4.x}, y{vec4.y}, z{vec4.z} {}

template <>
inline Vector3_t<float>::Vector3_t(const Vector2_t<float> &vec2)
    : x{vec2.x}, y{vec2.y}, z{0} {}

template <>
inline Vector3_t<int>::Vector3_t(const Vector4_t<int> &vec4)
    : x{vec4.x}, y{vec4.y}, z{vec4.z} {}

template <>
inline Vector3_t<int>::Vector3_t(const Vector2_t<int> &vec2)
    : x{vec2.x}, y{vec2.y}, z{0} {}

template <typename T>
inline T Vector3_t<T>::operator[](int i) const {
    assert(i >= 0 && i < 3);
    return (reinterpret_cast<const T *>(this))[i];
}

template <typename T>
inline T &Vector3_t<T>::operator[](int i) {
    assert(i >= 0 && i < 3);
    return (reinterpret_cast<T *>(this))[i];
}

template <typename T>
inline bool Vector3_t<T>::operator==(const Vector3_t<T> &v) const {
    return x == v.x && y == v.y && z == v.z;
}

template <typename T>
inline bool Vector3_t<T>::operator!=(const Vector3_t<T> &v) const {
    return !(*this == v);
}

template <typename T>
inline Vector3_t<T> Vector3_t<T>::operator-() const {
    return {-x, -y, -z};
}

template <typename T>
inline Vector3_t<T> &Vector3_t<T>::operator+=(const Vector3_t<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

template <typename T>
inline Vector3_t<T> &Vector3_t<T>::operator-=(const Vector3_t<T> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

template <typename T>
inline Vector3_t<T> &Vector3_t<T>::operator*=(const Vector3_t<T> &v) {
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
}

template <typename T>
inline Vector3_t<T> &Vector3_t<T>::operator/=(const Vector3_t<T> &v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
}

template <typename T>
inline Vector3_t<T> &Vector3_t<T>::operator*=(T s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
}

template <typename T>
inline Vector3_t<T> &Vector3_t<T>::operator/=(T s) {
    x /= s;
    y /= s;
    z /= s;
    return *this;
}

template <typename T>
inline float Vector3_t<T>::length() const {
    return sqrtf(x * x + y * y + z * z);
}

template <typename T>
inline float Vector3_t<T>::length_sqr() const {
    return x * x + y * y + z * z;
}

//
// Vector4
//

#ifdef MATHF_SSE
inline float4 operator+(const float4 &a, const float4 &b) {
    return float4{_mm_add_ps(a.m128, b.m128)};
}

inline float4 operator-(const float4 &a, const float4 &b) {
    return float4{_mm_sub_ps(a.m128, b.m128)};
}

inline float4 operator*(const float4 &a, const float4 &b) {
    return float4{_mm_mul_ps(a.m128, b.m128)};
}

inline float4 operator/(const float4 &a, const float4 &b) {
    return float4{_mm_div_ps(a.m128, b.m128)};
}

template <>
inline bool float4::operator==(const float4 &v) const {
    return (_mm_movemask_ps(_mm_cmpeq_ps(m128, v.m128)) & 0xf) == 0xf;
}

template <>
inline float4 float4::operator-() const {
    __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return float4{_mm_xor_ps(m128, mask)};
}

inline int4 operator+(const int4 &a, const int4 &b) {
    return int4{_mm_add_epi32(a.m128i, b.m128i)};
}

inline int4 operator-(const int4 &a, const int4 &b) {
    return int4{_mm_sub_epi32(a.m128i, b.m128i)};
}
#endif // MATHF_SSE

template <typename T>
inline Vector4_t<T> operator+(const Vector4_t<T> &a, const Vector4_t<T> &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

template <typename T>
inline Vector4_t<T> operator-(const Vector4_t<T> &a, const Vector4_t<T> &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

template <typename T>
inline Vector4_t<T> operator*(const Vector4_t<T> &a, const Vector4_t<T> &b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

template <typename T>
inline Vector4_t<T> operator/(const Vector4_t<T> &a, const Vector4_t<T> &b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

template <typename T>
inline Vector4_t<T> operator*(const Vector4_t<T> &v, float s) {
    return {v.x * s, v.y * s, v.z * s, v.w * s};
}

template <typename T>
inline Vector4_t<T> operator*(float s, const Vector4_t<T> &v) {
    return {s * v.x, s * v.y, s * v.z, s * v.w};
}

template <typename T>
inline Vector4_t<T> operator/(const Vector4_t<T> &v, float s) {
    return {v.x / s, v.y / s, v.z / s, v.w / s};
}

inline float4 operator/(const float4 &v, float s) {
    float invs = 1.0f / s;
    return {v.x * invs, v.y * invs, v.z * invs, v.w * invs};
}

template <>
inline Vector4_t<float>::Vector4_t(const Vector3_t<float> &vec3)
    : x{vec3.x}, y{vec3.y}, z{vec3.z}, w{0} {}

template <>
inline Vector4_t<float>::Vector4_t(const Vector2_t<float> &vec2)
    : x{vec2.x}, y{vec2.y}, z{0}, w{0} {}

template <>
inline Vector4_t<int>::Vector4_t(const Vector3_t<int> &vec3)
    : x{vec3.x}, y{vec3.y}, z{vec3.z}, w{0} {}

template <>
inline Vector4_t<int>::Vector4_t(const Vector2_t<int> &vec2)
    : x{vec2.x}, y{vec2.y}, z{0}, w{0} {}

template <typename T>
inline T Vector4_t<T>::operator[](int i) const {
    assert(i >= 0 && i < 4);
    return (reinterpret_cast<const T *>(this))[i];
}

template <typename T>
inline T &Vector4_t<T>::operator[](int i) {
    assert(i >= 0 && i < 4);
    return (reinterpret_cast<T *>(this))[i];
}

template <typename T>
inline bool Vector4_t<T>::operator==(const Vector4_t<T> &v) const {
    return x == v.x && y == v.y && z == v.z && w == v.w;
}

template <typename T>
inline bool Vector4_t<T>::operator!=(const Vector4_t<T> &v) const {
    return !(*this == v);
}

template <typename T>
inline Vector4_t<T> Vector4_t<T>::operator-() const {
    return {-x, -y, -z, -w};
}

template <typename T>
inline Vector4_t<T> &Vector4_t<T>::operator+=(const Vector4_t<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
}

template <typename T>
inline Vector4_t<T> &Vector4_t<T>::operator-=(const Vector4_t<T> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
}

template <typename T>
inline Vector4_t<T> &Vector4_t<T>::operator*=(const Vector4_t<T> &v) {
    x *= v.x;
    y *= v.y;
    z *= v.z;
    w *= v.w;
    return *this;
}

template <typename T>
inline Vector4_t<T> &Vector4_t<T>::operator/=(const Vector4_t<T> &v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
    w /= v.w;
    return *this;
}

template <typename T>
inline Vector4_t<T> &Vector4_t<T>::operator*=(T s) {
    x *= s;
    y *= s;
    z *= s;
    w *= s;
    return *this;
}

template <typename T>
inline Vector4_t<T> &Vector4_t<T>::operator/=(T s) {
    x /= s;
    y /= s;
    z /= s;
    w /= s;
    return *this;
}

template <typename T>
inline float Vector4_t<T>::length() const {
    return sqrtf(x * x + y * y + z * z + w * w);
}

template <typename T>
inline float Vector4_t<T>::length_sqr() const {
    return x * x + y * y + z * z + w * w;
}

} // internal

struct Rect {
    float x, y, w, h;

    Rect() {}
    Rect(float x, float y, float w, float h)
        : x{x}, y{y}, w{w}, h{h} {}

    Rect(const float2 &position, const float2 &size);

    static inline Rect zero();

    bool operator==(const Rect &rect) const;
    bool operator!=(const Rect &rect) const;

    bool contains(const float2 &v) const;
    bool overlaps(const Rect &rect) const;
};

union IntFloatUnion {
    uint32_t i32;
    float f;
};

union IntDoubleUnion {
    uint64_t i64;
    double d;
};

//
// Misc
//

inline float rsqrt(float value) {
    return 1.0f / sqrtf(value);
}

// Approximate reciprocal square root using SSE instructions
// when available.
inline float fast_rsqrt(float value) {
#ifdef MATHF_SSE
    __m128 result = _mm_rsqrt_ss(_mm_load_ss(&value));
    return *(reinterpret_cast<float *>(&result));
#else
    return rsqrt(value);
#endif
}

// Faster than int(floorf(value))
inline int floortoi(float value) {
    int x = int(value);
    return value < x ? x - 1 : x;
}

inline float clamp(float value, float a, float b) {
    // Both clang and MSVC will remove the branch using simd instructions
    if (value < a) return a;
    if (value > b) return b;
    return value;
}

inline int clampi(int value, int a, int b) {
    if (value < a) return a;
    if (value > b) return b;
    return value;
}

// Saturate function
inline float clamp01(float value) {
    return clamp(value, 0.0f, 1.0f);
}

// Interpolates between [a, b]
inline float lerp(float a, float b, float t) {
    return a + (b - a) * clamp01(t);
}

// Returns the position in the curve `t` for a given value `x`.
// The opposite of a lerp.
inline float unlerp(float a, float b, float x) {
    return (x - a) / (b - a);
}

// Maps a value x in range [a, b] to be in the same relative position
// in the range [c, d].
inline float remap(float a, float b, float c, float d, float x) {
    return lerp(c, d, unlerp(a, b, x));
}

// Returns 1 if value is positive, -1 if value is negative and 0 otherwise.
inline float signf(float value) {
    if (value > 0.0f) return 1.0f;
    if (value < 0.0f) return -1.0f;
    return 0.0f;
}

inline uint64_t next_pow2(uint64_t value) {
    --value;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return value + 1;
}

// Vector swizzle similar to GLSL or HLSL code
//
//     // HLSL
//     float2 a = float4(1.0, 2.0, 3.0, 4.0).zw;
//     float4 b = float2(1.0, 2.0).xxyy;
//
//     // shuffle
//     float2 a = shuffle<AxisZ, AxisW>(float4(1, 2, 3, 4));
//     float4 b = shuffle<AxisX, AxisX, AxisY, AxisY>(float2(1, 2));

#define TWO_M_SHUFFLE2(size_)                                                \
    template <int Index0, int Index1>                                        \
    inline float2 shuffle(const float##size_ &v) {                           \
        static_assert(Index0 >= 0 && Index0 >= 0                             \
                      && Index0 < size_ && Index1 < size_,                   \
                      "Index out of range");                                 \
        return {v[Index0], v[Index1]};                                       \
    }                                                                        \

TWO_M_SHUFFLE2(2);
TWO_M_SHUFFLE2(3);
TWO_M_SHUFFLE2(4);
#undef TWO_M_SHUFFLE2

#define TWO_M_SHUFFLE3(size_)                                                \
    template <int Index0, int Index1, int Index2>                            \
    inline float3 shuffle(const float##size_ &v) {                           \
        static_assert(Index0 >= 0 && Index0 >= 0 && Index2 >= 0              \
                      && Index0 < size_ && Index1 < size_ && Index2 < size_, \
                      "Index out of range");                                 \
        return {v[Index0], v[Index1], v[Index2]};                            \
    }                                                                        \

TWO_M_SHUFFLE3(2);
TWO_M_SHUFFLE3(3);
TWO_M_SHUFFLE3(4);
#undef TWO_M_SHUFFLE3

#define TWO_M_SHUFFLE4(size_)                                                \
    template <int Index0, int Index1, int Index2, int Index3>                \
    inline float4 shuffle(const float##size_ &v) {                           \
        static_assert(Index0 >= 0 && Index0 >= 0                             \
                      && Index2 >= 0 && Index3 >= 0                          \
                      && Index0 < size_ && Index1 < size_                    \
                      && Index2 < size_ && Index3 < size_,                   \
                      "Index out of range");                                 \
        return {v[Index0], v[Index1], v[Index2], v[Index3]};                 \
    }                                                                        \

TWO_M_SHUFFLE4(2);
TWO_M_SHUFFLE4(3);
TWO_M_SHUFFLE4(4);
#undef TWO_M_SHUFFLE4

//
// Rect
//

inline Rect::Rect(const float2 &position, const float2 &size) {
    x = position.x;
    y = position.y;
    w = size.x;
    h = size.y;
}

inline Rect Rect::zero() {
    return Rect{0.0f, 0.0f, 0.0f, 0.0f};
}

inline bool Rect::operator==(const Rect &rect) const {
    return x == rect.x && y == rect.y && w == rect.w && h == rect.h;
}

inline bool Rect::operator!=(const Rect &rect) const {
    return !(*this == rect);
}

inline bool Rect::contains(const float2 &v) const {
    return v.x >= x && v.y >= v.y && v.x < (w + x) && v.y < (h + y);
}

inline bool Rect::overlaps(const Rect &rect) const {
    return ((rect.x + rect.w) > x && rect.x < (x + w)
            && (rect.y + rect.h) > y && rect.y < (x + h));
}

inline void pprint(const Rect &rect) {
    printf("Rect(x: %f, y: %f, w: %f, h: %f)",
           rect.x, rect.y, rect.w, rect.h);
}

//
// float2
//

inline float dot(const float2 &a, const float2 &b) {
    return a.x * b.x + a.y * b.y;
}

inline float2 clamp(const float2 &v, float a, float b) {
    return float2{clamp(v.x, a, b),
                  clamp(v.y, a, b)};
}

inline float2 clamp01(const float2 &v) {
    return clamp(v, 0.0f, 1.0f);
}

inline float2 lerp(const float2 &a, const float2 &b, float t) {
    t = clamp01(t);
    return float2{a.x + (b.x - a.x) * t,
                  a.y + (b.y - a.y) * t};
}

inline float2 vstep(const float2 &edge, const float2 &v) {
    return float2(v.x >= edge.x, v.y >= edge.y);
}

inline float2 vmin(const float2 &a, const float2 &b) {
    return float2{fminf(a.x, b.x), fminf(a.y, b.y)};
}

inline float2 vmax(const float2 &a, const float2 &b) {
    return float2{fmaxf(a.x, b.x), fmaxf(a.y, b.y)};
}

inline float2 vabs(const float2 &v) {
    return float2{fabsf(v.x), fabsf(v.y)};
}

inline float2 vfloor(const float2 &v) {
    return float2{floorf(v.x), floorf(v.y)};
}

inline float2 vceil(const float2 &v) {
    return float2{ceilf(v.x), ceilf(v.y)};
}

inline float2 vsqrt(const float2 &v) {
    return float2{sqrtf(v.x), sqrtf(v.y)};
}

inline float2 vrsqrt(const float2 &v) {
    return float2{rsqrt(v.x), rsqrt(v.y)};
}

// Returns the angle in radians between two vectors.
inline float vangle_rad(const float2 &a, const float2 &b) {
    float denom = sqrtf(a.length_sqr() * b.length_sqr());
    if (denom < Epsilon * Epsilon) {
        return 0.0f;
    }
    return acosf(clamp(dot(a, b) / denom, -1.0f, 1.0f));
}

// Returns the angle in degrees between two vectors.
inline float vangle(const float2 &a, const float2 &b) {
    return vangle_rad(a, b) * RadToDeg;
}

inline float2 frac(const float2 &v) {
    return v - vfloor(v);
}

inline float2 normalize(const float2 &v) {
    return v / v.length();
}

inline float2 normalize_safe(const float2 &v) {
    float len = v.length();
    if (len == 0.0f) {
        return float2(0.0f);
    }
    return v / len;
}

inline float2 reflect(const float2 &v, const float2 &normal) {
    float s = 2.0f * dot(v, normal);
    return float2{v.x - s * normal.x,
                  v.y - s * normal.y};
}

inline void pprint(const float2 &v, const char *label = "") {
    printf("%s(%f, %f)\n", label, v.x, v.y);
}

//
// float3
//

inline float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float3 cross(const float3 &a, const float3 &b) {
    return float3{a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x};
}

inline float3 lerp(const float3 &a, const float3 &b, float t) {
    t = clamp01(t);
    return float3{a.x + (b.x - a.x) * t,
                  a.y + (b.y - a.y) * t,
                  a.z + (b.z - a.z) * t};
}

inline float3 clamp(const float3 &v, float a, float b) {
    return float3{clamp(v.x, a, b),
                  clamp(v.y, a, b),
                  clamp(v.z, a, b)};
}

inline float3 clamp01(const float3 &v) {
    return clamp(v, 0.0f, 1.0f);
}

inline float3 vstep(const float3 &edge, const float3 &v) {
    return float3(v.x >= edge.x, v.y >= edge.y, v.z >= edge.z);
}

inline float3 vmin(const float3 &a, const float3 &b) {
    return float3{fminf(a.x, b.x),
                  fminf(a.y, b.y),
                  fminf(a.z, b.z)};
}

inline float3 vmax(const float3 &a, const float3 &b) {
    return float3{fmaxf(a.x, b.x),
                  fmaxf(a.y, b.y),
                  fmaxf(a.z, b.z)};
}

inline float3 vabs(const float3 &v) {
    return float3{fabsf(v.x), fabsf(v.y), fabsf(v.z)};
}

inline float3 vfloor(const float3 &v) {
    return float3{floorf(v.x), floorf(v.y), floorf(v.z)};
}

inline float3 vceil(const float3 &v) {
    return float3{ceilf(v.x), ceilf(v.y), ceilf(v.z)};
}

inline float3 vsqrt(const float3 &v) {
    return float3{sqrtf(v.x), sqrtf(v.y), sqrtf(v.z)};
}

inline float3 vrsqrt(const float3 &v) {
    return float3{rsqrt(v.x), rsqrt(v.y), rsqrt(v.z)};
}

inline float3 frac(const float3 &v) {
    return v - vfloor(v);
}

inline float3 normalize(const float3 &v) {
    return v / v.length();
}

inline float3 normalize_safe(const float3 &v) {
    float len = v.length();
    if (len == 0.0f) {
        return float3(0.0f);
    }
    return v / len;
}

inline float3 reflect(const float3 &v, const float3 &normal) {
    float s = 2.0f * dot(v, normal);
    return float3{v.x - s * normal.x,
                  v.y - s * normal.y,
                  v.z - s * normal.z};
}

inline float3 refract(const float3 &v, const float3 &normal, float eta) {
    float3 paralell = eta * (v + dot(-v, normal) * normal);
    float3 perp = -sqrtf(1.0f - paralell.length_sqr()) * normal;
    return paralell + perp;
}

inline void pprint(const float3 &v, const char *label = "") {
    printf("%s(%f, %f, %f)\n", label, v.x, v.y, v.z);
}

//
// float4
//

inline float dot(const float4 &a, const float4 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline float4 lerp(const float4 &a, const float4 &b, float t) {
    t = clamp01(t);
    return float4{a.x + (b.x - a.x) * t,
                  a.y + (b.y - a.y) * t,
                  a.z + (b.z - a.z) * t,
                  a.w + (b.w - a.w) * t};
}

inline float4 clamp(const float4 &v, float a, float b) {
    return float4{clamp(v.x, a, b),
                  clamp(v.y, a, b),
                  clamp(v.z, a, b),
                  clamp(v.w, a, b)};
}

inline float4 clamp01(const float4 &v) {
    return clamp(v, 0.0f, 1.0f);
}

inline float4 vstep(const float4 &edge, const float4 &v) {
    return float4(v.x >= edge.x,
                  v.y >= edge.y,
                  v.z >= edge.z,
                  v.w >= edge.w);
}

inline float4 vmin(const float4 &a, const float4 &b) {
    return float4{fminf(a.x, b.x),
                  fminf(a.y, b.y),
                  fminf(a.z, b.z),
                  fminf(a.w, b.w)};
}

inline float4 vmax(const float4 &a, const float4 &b) {
    return float4{fmaxf(a.x, b.x),
                  fmaxf(a.y, b.y),
                  fmaxf(a.z, b.z),
                  fmaxf(a.w, b.w)};
}

inline float4 vabs(const float4 &v) {
    return float4{fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w)};
}

inline float4 vfloor(const float4 &v) {
    return float4{floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w)};
}

inline float4 vceil(const float4 &v) {
    return float4{ceilf(v.x), ceilf(v.y), ceilf(v.z), ceilf(v.w)};
}

inline float4 vsqrt(const float4 &v) {
#ifdef MATHF_SSE
    return _mm_sqrt_ps(v.m128);
#else
    return float4{sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), sqrtf(v.w)};
#endif
}

inline float4 vrsqrt(const float4 &v) {
#ifdef MATHF_SSE
    return _mm_rsqrt_ps(v.m128);
#else
    return float4{rsqrt(v.x), rsqrt(v.y), rsqrt(v.z), rsqrt(v.w)};
#endif
}

inline float4 frac(const float4 &v) {
    return v - vfloor(v);
}

inline float4 normalize(const float4 &v) {
    return v / v.length();
}

inline float4 normalize_safe(const float4 &v) {
    float len = v.length();
    if (len == 0.0f) {
        return float3(0.0f);
    }
    return v / len;
}

inline void pprint(const float4 &v, const char *label = "") {
    printf("%s(%f, %f, %f, %f)\n", label, v.x, v.y, v.z, v.w);
}

//
// int2
//

inline int2 clampi(const int2 &v, int a, int b) {
    return int2{clampi(v.x, a, b),
                clampi(v.y, a, b)};
}

inline int2 vabs(const int2 &v) {
    return int2{abs(v.x), abs(v.y)};
}

inline int2 vmin(const int2 &a, const int2 &b) {
    return int2{std::min(a.x, b.x),
                std::min(a.y, b.y)};
}

inline int2 vmax(const int2 &a, const int2 &b) {
    return int2{std::max(a.x, b.x),
                std::max(a.y, b.y)};
}

inline void pprint(const int2 &v, const char *label = "") {
    printf("%s(%d, %d)\n", label, v.x, v.y);
}

//
// int3
//

inline int3 clampi(const int3 &v, int a, int b) {
    return int3{clampi(v.x, a, b),
                clampi(v.y, a, b),
                clampi(v.z, a, b)};
}

inline int3 vabs(const int3 &v) {
    return int3{abs(v.x), abs(v.y), abs(v.z)};
}

inline int3 vmin(const int3 &a, const int3 &b) {
    return int3{std::min(a.x, b.x),
                std::min(a.y, b.y),
                std::min(a.z, b.z)};
}

inline int3 vmax(const int3 &a, const int3 &b) {
    return int3{std::max(a.x, b.x),
                std::max(a.y, b.y),
                std::max(a.z, b.z)};
}

inline void pprint(const int3 &v, const char *label = "") {
    printf("%s(%d, %d, %d)\n", label, v.x, v.y, v.z);
}

//
// int4
//

inline int4 clampi(const int4 &v, int a, int b) {
    return int4{clampi(v.x, a, b),
                clampi(v.y, a, b),
                clampi(v.z, a, b),
                clampi(v.w, a, b)};
}

inline int4 vabs(const int4 &v) {
    return int4{abs(v.x), abs(v.y), abs(v.z), abs(v.w)};
}

inline int4 vmin(const int4 &a, const int4 &b) {
    return int4{std::min(a.x, b.x),
                std::min(a.y, b.y),
                std::min(a.z, b.z),
                std::min(a.w, b.w)};
}

inline int4 vmax(const int4 &a, const int4 &b) {
    return int4{std::max(a.x, b.x),
                std::max(a.y, b.y),
                std::max(a.z, b.z),
                std::max(a.w, b.w)};
}

inline void pprint(const int4 &v, const char *label = "") {
    printf("%s(%d, %d, %d, %d)\n", label, v.x, v.y, v.z, v.w);
}

} // mathf

#endif // MATHF_H
