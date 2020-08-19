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

#ifndef MATHF_NOISE_H
#define MATHF_NOISE_H

#include <cstdint>
#include <cassert>

#include "mathf.h"

namespace mathf {

// Implementation of xorshift*. Xorshift* has low linear complexity in
// the lower bits which are discarded when generating floats and 32 bit
// intergers. Uses a single 64bit integer for the state so it is cheap
// to copy this struct.
struct Xorshift64 {
    uint64_t state;

    // Seed value chosen completly at random
    Xorshift64(uint64_t seed = 5);

    // Returns a uniformly random int64
    uint64_t randi64();

    // Returns a uniformly random int32
    uint32_t randi();

    // Returns a uniformly random double in the range [0, 1)
    double randf64();

    // Returns a uniformly random float in the range [0, 1)
    float randf();

    // Returns a random vector with all components in the range [0, 1)
    float2 rand2();

    // Returns a random vector with all components in the range [0, 1)
    float3 rand3();

    // Returns a random vector with all components in the range [0, 1)
    float4 rand4();

    // Returns a uniformly random double in the range [a, b)
    double randf64(double a, double b);

    // Returns a uniformly random float in the range [a, b)
    float randf(float a, float b);

    // Returns a random vector with all components in the range [a, b)
    float2 rand2(const float2 &a, const float2 &b);

    // Returns a random vector with all components in the range [a, b)
    float3 rand3(const float3 &a, const float3 &b);

    // Returns a random vector with all components in the range [a, b)
    float4 rand4(const float4 &a, const float4 &b);

    // Returns a random 2D vector with length less than 1
    float2 in_unit_circle();

    // Returns a random 3D vector with length less than 1
    float3 in_unit_sphere();
};

// Used to initialize the xorshift64 state
struct SplitMix64 {
    uint64_t state;

    SplitMix64(uint64_t seed) : state{seed} {}

    uint64_t randi64();
};

// The default random number generator
using Random = Xorshift64;

// 2D simplex noise.
float snoise(const float2 &v);

// 3D simplex noise.
float snoise(const float3 &v);

// 4D simplex noise.
float snoise(const float4 &v);

// 2D simplex fractal noise.
// Noise frequency is multiplied by `lacunarity` for each octave
// `gain` controls how much each octave contributes to the final output.
float snoise_fractal(float2 v, int octaves, float lacunarity, float gain);

// 3D simplex fractal noise.
// Noise frequency is multiplied by `lacunarity` for each octave
// `gain` controls how much each octave contributes to the final output.
float snoise_fractal(float3 v, int octaves, float lacunarity, float gain);

// 3D simplex billow fractal noise.
float snoise_fractal_b(float2 v, int octaves, float lacunarity, float gain);

// 2D simplex billow fractal noise.
float snoise_fractal_b(float3 v, int octaves, float lacunarity, float gain);

// Sets the seed for the simplex noise generator functions by shuffling
// values in the permutation table.
void snoise_seed(uint64_t seed);

inline uint64_t SplitMix64::randi64() {
    uint64_t s = state;
    state = s + 0x9e3779b97f4a7c15;
    s = (s ^ (s >> 30)) * 0xbf58476d1ce4e5b9;
    s = (s ^ (s >> 27)) * 0x94d049bb133111eb;
    return s ^ (s >> 31);
}

inline Xorshift64::Xorshift64(uint64_t seed) {
    SplitMix64 sm{seed};
    this->state = sm.randi64();
}

inline uint64_t Xorshift64::randi64() {
    assert(state != 0 && "Xorshift: Invalid state");
    uint64_t s = state;
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    state = s;
    return s * UINT64_C(0x2545f4914f6cdd1d);
}

inline uint32_t Xorshift64::randi() {
    return randi64() >> 32;
}

inline double Xorshift64::randf64() {
    // (randi64() >> 11) * 0x1.0p-53
    IntDoubleUnion u {.i64 = (UINT64_C(0x3ff) << 52) | (randi64() >> 12)};
    return u.d - 1.0;
}

inline float Xorshift64::randf() {
    IntFloatUnion u = {.i32 = 0x3f800000 | (randi() >> 9)};
    return u.f - 1.0f;
}

inline float2 Xorshift64::rand2() {
    return float2{randf(), randf()};
}

inline float3 Xorshift64::rand3() {
    return float3{randf(), randf(), randf()};
}

inline float4 Xorshift64::rand4() {
    return float4{randf(), randf(), randf(), randf()};
}

inline float Xorshift64::randf(float a, float b) {
    return randf() * (b - a) + a;
}

inline float2 Xorshift64::rand2(const float2 &a, const float2 &b) {
    return rand2() * (b - a) + a;
}

inline float3 Xorshift64::rand3(const float3 &a, const float3 &b) {
    return rand3() * (b - a) + a;
}

inline float4 Xorshift64::rand4(const float4 &a, const float4 &b) {
    return rand4() * (b - a) + a;
}

inline float2 Xorshift64::in_unit_circle() {
    for (;;) {
        float2 r{randf(-1.0f, 1.0f),
                 randf(-1.0f, 1.0f)};
        if (r.length_sqr() < 1.0f) return r;
    }
}

inline float3 Xorshift64::in_unit_sphere() {
    for (;;) {
        float3 r{randf(-1.0f, 1.0f),
                 randf(-1.0f, 1.0f),
                 randf(-1.0f, 1.0f)};
        if (r.length_sqr() < 1.0f) return r;
    }
}

} // mathf

#endif // MATHF_NOISE_H
