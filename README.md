# mathf

Small C++ library with vector math routines, Xorshift random number generator, and simplex noise.

## Misc math

- `rsqrt(x)` Returns 1 / sqrtf(x)
- `fast_rsqrt(x)` Returns the approximate reciprocal square root using SSE intrinsics when available
- `floortoi(x)` Floor a float converting it to an int. This is faster than `static_cast<int>(floorf(x))`
- `clamp(x, a, b)` Clamp a value to be in the the range [a, b]
- `clampi(x, a, b)` Same as clamp but for integers instead of floats
- `clamp01(x)` The saturate function, clamps a value to be in [0, 1]
- `lerp(a, b, t)` Linearly interpolate between vector a and b at time t where t is in [0, 1]
- `unlerp(a, b, x)` The opposite of `lerp`, returns the position in the curve `t` for a given value `x`.
- `remap(a, b, c, d, x)` Maps a value x in the range [a, b] to be in the range [c, d]
- `signf(x)` returns 1 if x is positive, -1 if x is negative, 0 otherwise.
- `next_pow2(x)` Returns the next power of two given a unsigned integer x.

## Vector operators

- `-vec`
- `vec + vec`, `vec - vec`
- `vec * vec, vec / vec` Component-wise multiply and divide
- `vec * scalar` `scalar * vec`, `vec / scalar`
- `+=`, `-=`, `*=`, `/=`
- `vec(scalar)` Initializes all components to a scalar value, can be used to represent the zero vector as `float3(0)` for example. This is an explicit constructor, so scalars will not implicitly convert to vector types. Similar constructors exist to convert between vectors of different types/dimensions.

For 4D vectors, most operators map to their equivalent SSE instructions.

## Vector member functions

- `length()` Returns length / magnitude / norm of the vector.
- `length_sqr()` Returns the length of the vector squared.

## Float vectors (float2, float3, float4)

Functions defined for all floating point vector types.

- `dot(a, b)`
- `normalize(v)`
- `normalize_safe(v)` Returns a normalized vector without producing NaN when v is the zero vector.
- `clamp(v, a, b)` Clamps each vector component to be in the range [a, b]
- `clamp01(v)` Saturate function, clamps the vector have all components in [0, 1]
- `lerp(a, b, t)` Linearly interpolate between vector a and b at time t where t is in [0, 1]
- `frac(v)` Returns `1 - vfloor(v)`
- `vstep(edge, v)` Returns a vector with each component equal to `v[i] >= edge[i]`
- `vmin(a, b)` Component-wise `fminf`
- `vmax(a, b)` Component-wise `fmaxf`
- `vabs(a, b)` Component-wise `fabsf`
- `vfloor(a, b)` Component-wise `floorf`
- `vceil(a, b)` Component-wise `ceilf`
- `vsqrt(a, b)` Component-wise `ceilf`
- `vrsqrt(a, b)` Component-wise `rsqrt`, for `float4` the `rsqrt` SSE instrinsic is used.
- `pprint(v)` Print a vector

### float2 specific functions

- `vangle(a, b)` Returns the angle between two vectors in degrees
- `vangle_rad(a, b)` Returns the angle between two vectors in radians
- `reflect(v, normal)`

### float3 specific functions

- `cross(a, b)`
- `reflect(v, normal)`
- `refract(v, normal, eta)`

## Integer vectors (int2, int3, int4)

- `clampi(v, a, b)`
- `vabs(v)`
- `vmin(a, b)`
- `vmax(a, b)`

## Vector shuffles (swizzle)

Constructs a new vector by selecting elements from a given vector.

```cpp
shuffle<0, 0, 1, 1>(float2(1, 2)) // float4(1, 1, 2, 2)
shuffle<3, 2, 1, 0>(float4(1, 2, 3, 4)) // float4(4, 3, 2, 1)
shuffle<1, 2>(float3(1, 2, 3)) // float2(2, 3)
```

## Vector SIMD

The 4D vector types (int4 and float4) use sse instructions when available for most operators. To use `vec.m128` and `vec.m128i` to use with float and integer sse instructions respectively. `__m128` can be implicitly be converted to `float4` and `__m128i` to `int4`.
