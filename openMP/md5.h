#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h> 

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
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

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
// 定义了一系列MD5中的具体函数
// 这四个计算函数是需要你进行SIMD并行化的
// 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))
// 并行化的MD5基本函数（四路并行）
inline uint32x4_t F_NEON(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
  return vorrq_u32(vandq_u32(x, y), vandq_u32(vmvnq_u32(x), z));
}

inline uint32x4_t G_NEON(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
  return vorrq_u32(vandq_u32(x, z), vandq_u32(y, vmvnq_u32(z)));
}

inline uint32x4_t H_NEON(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
  return veorq_u32(veorq_u32(x, y), z);
}

inline uint32x4_t I_NEON(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
  return veorq_u32(y, vorrq_u32(x, vmvnq_u32(z)));
}
/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
// 定义了一系列MD5中的具体函数
// 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
// 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))

inline uint32x4_t ROTATELEFT_NEON(uint32x4_t num, const int n) {
  return vorrq_u32(vshlq_n_u32(num, n), vshrq_n_u32(num, 32 - n));
}

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

// 并行化的MD5轮函数
#define FF_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, F_NEON(b, c, d)); \
  a = vaddq_u32(a, x); \
  a = vaddq_u32(a, vdupq_n_u32(ac)); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

#define GG_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, G_NEON(b, c, d)); \
  a = vaddq_u32(a, x); \
  a = vaddq_u32(a, vdupq_n_u32(ac)); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

#define HH_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, H_NEON(b, c, d)); \
  a = vaddq_u32(a, x); \
  a = vaddq_u32(a, vdupq_n_u32(ac)); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

#define II_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, I_NEON(b, c, d)); \
  a = vaddq_u32(a, x); \
  a = vaddq_u32(a, vdupq_n_u32(ac)); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

// 四路并行MD5哈希函数
void MD5Hash_NEON(string inputs[4], bit32 states[4][4]);

void MD5Hash(string input, bit32 *state);