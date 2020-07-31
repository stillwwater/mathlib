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

#include "noise.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#include "config.h"
#include "mathf.h"

#ifdef MATHF_SSE
#include <xmmintrin.h>
#endif

namespace mathf {

static const float3 grad3[] = {
    {1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0},
    {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
    {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1},
};

static const float4 grad4[] = {
    { 0,  1, 1, 1}, { 0,  1,  1, -1}, { 0,  1, -1, 1}, { 0,  1, -1, -1},
    { 0, -1, 1, 1}, { 0, -1,  1, -1}, { 0, -1, -1, 1}, { 0, -1, -1, -1},
    { 1,  0, 1, 1}, { 1,  0,  1, -1}, { 1,  0, -1, 1}, { 1,  0, -1, -1},
    {-1,  0, 1, 1}, {-1,  0,  1, -1}, {-1,  0, -1, 1}, {-1,  0, -1, -1},
    { 1,  1, 0, 1}, { 1,  1,  0, -1}, { 1, -1,  0, 1}, { 1, -1,  0, -1},
    {-1,  1, 0, 1}, {-1,  1,  0, -1}, {-1, -1,  0, 1}, {-1, -1,  0, -1},
    { 1,  1, 1, 0}, { 1,  1, -1,  0}, { 1, -1,  1, 0}, { 1, -1, -1,  0},
    {-1,  1, 1, 0}, {-1,  1, -1,  0}, {-1, -1,  1, 0}, {-1, -1, -1,  0},
};

static unsigned char perm[512] = {
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
    184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
    222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,

    // And again so it wraps nicely
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
    184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
    222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180
};

template <typename T, typename U>
static inline float gradient(const U *grad, int gi, const T &v, float c) {
    float t = c - v.length_sqr();
    if (t < 0.0f) return 0.0f;
    t *= t;
    return t * t * dot(T(grad[gi]), v);
}

void snoise_seed(uint64_t seed) {
    Xorshift64 rng{seed};
    int len = sizeof(perm) / 2;
    for (int i = len - 1; i > 0; --i) {
        int target = rng.randf(0, i + 1);
        std::swap(perm[i], perm[target]);
    }
    // Duplicate for the other 256 elements
    memcpy(perm + len, perm, len);
}

// Based on the 2D simplex noise code by Stefan Gustavson.
float snoise(const float2 &v) {
    constexpr float F2 = 0.366025403784439f;  // 0.5*(sqrt(3.0)-1.0) (skew)
    constexpr float G2 = 0.211324865405187f;  // (3.0-sqrt(3.0))/6.0 (unskew)

    // Skew the input space to determine which simplex cell we're in
    float s = (v.x + v.y) * F2; // Hairy factor for 2D

    // First corner
    int2 i{floortoi(v.x + s), floortoi(v.y + s)};
    float2 t = float2((i.x + i.y) * G2);
    float2 x0 = v - (float2(i) - t);

    // For the 2D case, the simplex shape is an equilateral triangle.
    // Determine which simplex we are in.
    int2 i1;
    if (x0.x > x0.y)
        i1 = {1, 0};
    else
        i1 = {0, 1};

    // Other corners
    float2 x1 = x0 - float2(i1) + float2(G2);
    float2 x2 = x0 - float2(1.0) + float2(2.0 * G2);

    // Permutations
    int2 h{i.x & 255, i.y & 255};
    int gi0 = perm[h.x + perm[h.y]] % 12;
    int gi1 = perm[h.x + i1.x + perm[h.y + i1.y]] % 12;
    int gi2 = perm[h.x + 1 + perm[h.y + 1]] % 12;

    // Calculate the contribution from the three corners
    float n0 = gradient(grad3, gi0, x0, 0.5f);
    float n1 = gradient(grad3, gi1, x1, 0.5f);
    float n2 = gradient(grad3, gi2, x2, 0.5f);
    // Add contributions from each corner to get the final noise value.
    // The result is scaled to return values in the interval [-1,1].
    return 70.0 * (n0 + n1 + n2);
}

// Based on the 3D simplex noise code by Stefan Gustavson.
float snoise(const float3 &v) {
    constexpr float F3 = 1.0f / 3.0f; // (skew)
    constexpr float G3 = 1.0f / 6.0f; // (unskew)

    // Skew the input space to determine which simplex cell we're in
    float s = (v.x + v.y + v.z) * F3;

    // First corner
    int3 i{floortoi(v.x + s), floortoi(v.y + s), floortoi(v.z + s)};
    float3 t = float3((i.x + i.y + i.z) * G3);
    float3 x0 = v - (float3(i) - t);

    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    int3 i1, i2;
    if(x0.x >= x0.y) {
        if(x0.y >= x0.z) {
            i1 = {1, 0, 0};
            i2 = {1, 1, 0};
        } else if(x0.x >= x0.z) {
            i1 = {1, 0, 0};
            i2 = {1, 0, 1};
        } else {
            i1 = {0, 0, 1};
            i2 = {1, 0, 1};
        }
    } else {
        if(x0.y < x0.z) {
            i1 = {0, 0, 1};
            i2 = {0, 1, 1};
        } else if(x0.x < x0.z) {
            i1 = {0, 1, 0};
            i2 = {0, 1, 1};
        } else {
            i1 = {0, 1, 0};
            i2 = {1, 1, 0};
        }
    }
    // Other corners
    float3 x1 = x0 - float3(i1) + float3(G3);
    float3 x2 = x0 - float3(i2) + float3(2.0f * G3);
    float3 x3 = x0 - float3(1.0) + float3(3.0f * G3);

    // Permutations
    int3 h{i.x & 255, i.y & 255, i.z & 255};
    int gi0 = perm[h.x + perm[h.y + perm[h.z]]] % 12;
    int gi1 = perm[h.x + i1.x + perm[h.y + i1.y + perm[h.z + i1.z]]] % 12;
    int gi2 = perm[h.x + i2.x + perm[h.y + i2.y + perm[h.z + i2.z]]] % 12;
    int gi3 = perm[h.x + 1 + perm[h.y + 1 + perm[h.z + 1]]] % 12;

    // Contribution from the four corners
    float n0 = gradient(grad3, gi0, x0, 0.6f);
    float n1 = gradient(grad3, gi1, x1, 0.6f);
    float n2 = gradient(grad3, gi2, x2, 0.6f);
    float n3 = gradient(grad3, gi3, x3, 0.6f);

    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return 32.0f * (n0 + n1 + n2 + n3);
}

// Based on the improved simplex rank ordering method for 4D simplex noise
// by Stefan Gustavson.
float snoise(const float4 &v) {
    constexpr float F4 = 0.309016994375947f; // (sqrt(5.0)-1)/4.0 (skew)
    constexpr float G4 = 0.138196601125011f; // (5.0-sqrt(5.0))/20.0 (unskew)

    // Skew the space to determine which cell of 24 simplices we're in
    float s = (v.x + v.y + v.z + v.w) * F4;

    // First corner
    int4 i = {floortoi(v.x + s), floortoi(v.y + s),
              floortoi(v.z + s), floortoi(v.w + s)};

    float4 t = float4((i.x + i.y + i.z + i.w) * G4);
    float4 x0 = v - (float4(i) - t);

    // For the 4D case, the simplex shape is a pentatope instead of
    // a tesseract which is used in classic noise, meaning we only need to
    // calculate 5 vertecies instead of 16. For n dimentions, simplex noise
    // uses n+1 vertecies for the simplex shape, compared to classic noise
    // which requires n^2 vertecies.
    int4 rank = int4(0);
    if (x0.x > x0.y) rank.x++; else rank.y++;
    if (x0.x > x0.z) rank.x++; else rank.z++;
    if (x0.x > x0.w) rank.x++; else rank.w++;
    if (x0.y > x0.z) rank.y++; else rank.z++;
    if (x0.y > x0.w) rank.y++; else rank.w++;
    if (x0.z > x0.w) rank.z++; else rank.w++;

#ifdef MATHF_SSE
    // Since there is no sse cmpge instruction for integer vectors
    // we use cmplt and flip all the the bits.
    __m128i m1 = _mm_set1_epi32(1);
    int4 i1 = _mm_cmplt_epi32(rank.m128i, _mm_set1_epi32(3));
    i1.m128i = _mm_xor_si128(_mm_and_si128(i1.m128i, m1), m1);

    int4 i2 = _mm_cmplt_epi32(rank.m128i, _mm_set1_epi32(2));
    i2.m128i = _mm_xor_si128(_mm_and_si128(i2.m128i, m1), m1);

    int4 i3 = _mm_cmplt_epi32(rank.m128i, m1);
    i3.m128i = _mm_xor_si128(_mm_and_si128(i3.m128i, m1), m1);
#else
    // rank is a vector with the numbers 0, 1, 2 and 3 in some order.
    // We use a thresholding to set the coordinates in turn.
    int4 i1 = {rank.x >= 3, rank.y >= 3, rank.z >= 3, rank.w >= 3};
    int4 i2 = {rank.x >= 2, rank.y >= 2, rank.z >= 2, rank.w >= 2};
    int4 i3 = {rank.x >= 1, rank.y >= 1, rank.z >= 1, rank.w >= 1};
#endif

    // Other corners
    float4 x1 = x0 - float4(i1) + float4(G4);
    float4 x2 = x0 - float4(i2) + float4(2.0f * G4);
    float4 x3 = x0 - float4(i3) + float4(3.0f * G4);
    float4 x4 = x0 - float4(1.0f) + float4(4.0f * G4);

    // Permutations
    int4 h{i.x & 255, i.y & 255, i.z & 255, i.w & 255};
    int gi0 = perm[h.x + perm[h.y + perm[h.z + perm[h.w]]]] % 32;

    int gi1 = perm[h.x + i1.x + perm[h.y + i1.y
            + perm[h.z + i1.z + perm[h.w + i1.w]]]] % 32;

    int gi2 = perm[h.x + i2.x + perm[h.y + i2.y
            + perm[h.z + i2.z + perm[h.w + i2.w]]]] % 32;

    int gi3 = perm[h.x + i3.x + perm[h.y + i3.y
            + perm[h.z + i3.z + perm[h.w + i3.w]]]] % 32;

    int gi4 = perm[h.x + 1 + perm[h.y + 1
            + perm[h.z + 1 + perm[h.w + 1]]]] % 32;

    // Contribution from the four corners
    float n0 = gradient(grad4, gi0, x0, 0.6f);
    float n1 = gradient(grad4, gi1, x1, 0.6f);
    float n2 = gradient(grad4, gi2, x2, 0.6f);
    float n3 = gradient(grad4, gi3, x3, 0.6f);
    float n4 = gradient(grad4, gi4, x4, 0.6f);

    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return 27.0f * (n0 + n1 + n2 + n3 + n4);
}

template <typename T>
static inline float fractal_fbm(T v, int octaves, float lac, float gain) {
    float sum = 0.0f;
    float amp = 1.0f;
    float frac_range = 0.0f;

    for (int i = 0; i < octaves; ++i) {
        sum += snoise(v) * amp;
        frac_range += amp;
        amp *= gain;
        v *= lac;
    }
    return sum / frac_range;
}

template <typename T>
static inline float fractal_b(T v, int octaves, float lac, float gain) {
    float sum = 0.0f;
    float amp = 1.0f;
    float frac_range = 0.0f;

    for (int i = 0; i < octaves; ++i) {
        sum += (fabsf(snoise(v)) * 2.0f - 1.0f) * amp;
        frac_range += amp;
        amp *= gain;
        v *= lac;
    }
    return sum / frac_range;
}

float snoise_fractal(float3 v, int octaves, float lacunarity, float gain) {
    return fractal_fbm(v, octaves, lacunarity, gain);
}

float snoise_fractal(float2 v, int octaves, float lacunarity, float gain) {
    return fractal_fbm(v, octaves, lacunarity, gain);
}

float snoise_fractal_b(float3 v, int octaves, float lacunarity, float gain) {
    return fractal_b(v, octaves, lacunarity, gain);
}

float snoise_fractal_b(float2 v, int octaves, float lacunarity, float gain) {
    return fractal_b(v, octaves, lacunarity, gain);
}

} // mathf
