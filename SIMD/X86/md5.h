#include <iostream>
#include <string>
#include <cstring>
#include <emmintrin.h>  // SSE2
#include <tmmintrin.h>  // SSSE3

using namespace std;

typedef unsigned char Byte;
typedef unsigned int bit32;

// MD5 constants remain the same
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

// Basic MD5 functions for scalar version remain the same
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

// SSE versions of basic MD5 functions
inline __m128i F_SSE(__m128i x, __m128i y, __m128i z) {
    return _mm_or_si128(_mm_and_si128(x, y), _mm_andnot_si128(x, z));
}

inline __m128i G_SSE(__m128i x, __m128i y, __m128i z) {
    return _mm_or_si128(_mm_and_si128(x, z), _mm_andnot_si128(z, y));
}

inline __m128i H_SSE(__m128i x, __m128i y, __m128i z) {
    return _mm_xor_si128(_mm_xor_si128(x, y), z);
}

inline __m128i I_SSE(__m128i x, __m128i y, __m128i z) {
    return _mm_xor_si128(y, _mm_or_si128(x, _mm_andnot_si128(z, _mm_set1_epi32(-1))));
}

// Rotate left for scalar version
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))

// SSE version of rotate left
inline __m128i ROTATELEFT_SSE(__m128i num, const int n) {
    return _mm_or_si128(_mm_slli_epi32(num, n), _mm_srli_epi32(num, 32 - n));
}

// Scalar MD5 round functions remain the same
#define FF(a, b, c, d, x, s, ac) { \
  (a) += F ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define GG(a, b, c, d, x, s, ac) { \
  (a) += G ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define HH(a, b, c, d, x, s, ac) { \
  (a) += H ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define II(a, b, c, d, x, s, ac) { \
  (a) += I ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

// SSE versions of MD5 round functions
#define FF_SSE(a, b, c, d, x, s, ac) { \
  a = _mm_add_epi32(a, F_SSE(b, c, d)); \
  a = _mm_add_epi32(a, x); \
  a = _mm_add_epi32(a, _mm_set1_epi32(ac)); \
  a = ROTATELEFT_SSE(a, s); \
  a = _mm_add_epi32(a, b); \
}

#define GG_SSE(a, b, c, d, x, s, ac) { \
  a = _mm_add_epi32(a, G_SSE(b, c, d)); \
  a = _mm_add_epi32(a, x); \
  a = _mm_add_epi32(a, _mm_set1_epi32(ac)); \
  a = ROTATELEFT_SSE(a, s); \
  a = _mm_add_epi32(a, b); \
}

#define HH_SSE(a, b, c, d, x, s, ac) { \
  a = _mm_add_epi32(a, H_SSE(b, c, d)); \
  a = _mm_add_epi32(a, x); \
  a = _mm_add_epi32(a, _mm_set1_epi32(ac)); \
  a = ROTATELEFT_SSE(a, s); \
  a = _mm_add_epi32(a, b); \
}

#define II_SSE(a, b, c, d, x, s, ac) { \
  a = _mm_add_epi32(a, I_SSE(b, c, d)); \
  a = _mm_add_epi32(a, x); \
  a = _mm_add_epi32(a, _mm_set1_epi32(ac)); \
  a = ROTATELEFT_SSE(a, s); \
  a = _mm_add_epi32(a, b); \
}

// Function declarations remain the same
void MD5Hash_SSE(string inputs[4], bit32 states[4][4]);
void MD5Hash(string input, bit32 *state);