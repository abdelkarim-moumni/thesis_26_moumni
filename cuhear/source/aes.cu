#include "aes.hpp"

#include <cstring>
#include <wmmintrin.h>

#define INT_EXTRACT_0(x) ((x & 0xFF000000) >> 24)
#define INT_EXTRACT_1(x) ((x & 0x00FF0000) >> 16)
#define INT_EXTRACT_2(x) ((x & 0x0000FF00) >> 8)
#define INT_EXTRACT_3(x)  (x & 0x000000FF)

#define INT_UPDATE_0(x, u) x = (x & ~0xFF000000) | ((u) << 24)
#define INT_UPDATE_1(x, u) x = (x & ~0x00FF0000) | ((u) << 16)
#define INT_UPDATE_2(x, u) x = (x & ~0x0000FF00) | ((u) << 8)
#define INT_UPDATE_3(x, u) x = ((x & ~0x000000FF) | (u))

// Useful resource: https://legacy.cryptool.org/en/cto/aes-step-by-step

namespace aes {
    __constant__ uint8_t sbox[256] = {
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
    };

    static inline __device__ void ExpandKey(aes::u128 *key, aes::u128 *roundKeys) {

    }

    static inline __device__ void ShiftRows(aes::u128 *state) {
        uint8_t rot;

        rot = INT_EXTRACT_1(state->x);
        INT_UPDATE_1(state->x, INT_EXTRACT_1(state->y));
        INT_UPDATE_1(state->y, INT_EXTRACT_1(state->z));
        INT_UPDATE_1(state->z, INT_EXTRACT_1(state->w));
        INT_UPDATE_1(state->w, rot);

        rot = INT_EXTRACT_2(state->x);
        INT_UPDATE_2(state->x, INT_EXTRACT_2(state->z));
        INT_UPDATE_2(state->z, rot);

        rot = INT_EXTRACT_2(state->y);
        INT_UPDATE_2(state->y, INT_EXTRACT_2(state->w));
        INT_UPDATE_2(state->w, rot);

        rot = INT_EXTRACT_3(state->x);
        INT_UPDATE_3(state->x, INT_EXTRACT_3(state->w));
        INT_UPDATE_3(state->w, INT_EXTRACT_3(state->z));
        INT_UPDATE_3(state->z, INT_EXTRACT_3(state->y));
        INT_UPDATE_3(state->y, rot);
    }

    static inline __device__ void SubBytes(aes::u128 *state) {
        INT_UPDATE_0(state->x, sbox[INT_EXTRACT_0(state->x)]);
        INT_UPDATE_1(state->x, sbox[INT_EXTRACT_1(state->x)]);
        INT_UPDATE_2(state->x, sbox[INT_EXTRACT_2(state->x)]);
        INT_UPDATE_3(state->x, sbox[INT_EXTRACT_3(state->x)]);

        INT_UPDATE_0(state->y, sbox[INT_EXTRACT_0(state->y)]);
        INT_UPDATE_1(state->y, sbox[INT_EXTRACT_1(state->y)]);
        INT_UPDATE_2(state->y, sbox[INT_EXTRACT_2(state->y)]);
        INT_UPDATE_3(state->y, sbox[INT_EXTRACT_3(state->y)]);

        INT_UPDATE_0(state->z, sbox[INT_EXTRACT_0(state->z)]);
        INT_UPDATE_1(state->z, sbox[INT_EXTRACT_1(state->z)]);
        INT_UPDATE_2(state->z, sbox[INT_EXTRACT_2(state->z)]);
        INT_UPDATE_3(state->z, sbox[INT_EXTRACT_3(state->z)]);

        INT_UPDATE_0(state->w, sbox[INT_EXTRACT_0(state->w)]);
        INT_UPDATE_1(state->w, sbox[INT_EXTRACT_1(state->w)]);
        INT_UPDATE_2(state->w, sbox[INT_EXTRACT_2(state->w)]);
        INT_UPDATE_3(state->w, sbox[INT_EXTRACT_3(state->w)]);
    }

    #define XTIME(x) ((((x) << 1) ^ ((((x) >> 7) & 1) * 27)) & 0xFF)

    static inline __device__ void MixColumns(aes::u128 *state) {
        for (int i = 0; i < 4; i++) {
            uint32_t *column = (uint32_t*) state + i;
            uint8_t col0 = INT_EXTRACT_0(*column);
            uint8_t base = col0 ^ INT_EXTRACT_1(*column) ^ INT_EXTRACT_2(*column) ^ INT_EXTRACT_3(*column);

            INT_UPDATE_0(*column, col0 ^ XTIME(col0 ^ INT_EXTRACT_1(*column)) ^ base);
            INT_UPDATE_1(*column, INT_EXTRACT_1(*column) ^ XTIME(INT_EXTRACT_1(*column) ^ INT_EXTRACT_2(*column)) ^ base);
            INT_UPDATE_2(*column, INT_EXTRACT_2(*column) ^ XTIME(INT_EXTRACT_2(*column) ^ INT_EXTRACT_3(*column)) ^ base);
            INT_UPDATE_3(*column, INT_EXTRACT_3(*column) ^ XTIME(col0 ^ INT_EXTRACT_3(*column)) ^ base);
        }
    }

    static inline __device__ void AddRoundKey(aes::u128 *state, aes::u128 *roundKey) {
        state->x ^= roundKey->x;
        state->y ^= roundKey->y;
        state->z ^= roundKey->z;
        state->w ^= roundKey->w;
    }

    static inline __device__ void Run(aes::u128 *state, aes::u128 *roundKeys) {
        aes::AddRoundKey(state, roundKeys++);
        // First n-1 rounds
        for (int i = 0; i < aes::NUM_ROUNDS - 1; i++) {
            aes::SubBytes(state);
            aes::ShiftRows(state);
            aes::MixColumns(state);
            aes::AddRoundKey(state, roundKeys++);
        }
        // Last round
        aes::SubBytes(state);
        aes::ShiftRows(state);
        aes::AddRoundKey(state, roundKeys);
    }
}

namespace cuhear::rng {

    AesContext::AesContext(aes::u128 *roundKeys) {
        std::memcpy(this->roundKeys, roundKeys, sizeof(aes::u128) * (aes::NUM_ROUNDS + 1));
        // Swap endianness for 32bit values
        uint32_t *keys = reinterpret_cast<uint32_t *>(this->roundKeys);
        for (int i = 0; i < (aes::NUM_ROUNDS + 1) * 4; i++) {
            uint32_t key = keys[i];
            keys[i] = ((key >> 24) & 0xFF) | ((key << 8) & 0xFF0000) | ((key >> 8) & 0xFF00) | ((key << 24) & 0xFF000000);
        }
    }

    __device__ AesContext AesContext::ExpandOnDevice(aes::u128 key) {
        aes::u128 roundKeys[aes::NUM_ROUNDS + 1];
        aes::ExpandKey(&key, roundKeys);
        return { roundKeys };
    }

    #define AESNI128_KEY_EXPAND(k, rcon) (aesni128_key_expand(k, _mm_aeskeygenassist_si128(k, rcon)))

    static __m128i aesni128_key_expand(__m128i key, __m128i keygened) {
        keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3,3,3,3));
        key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
        key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
        key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
        return _mm_xor_si128(key, keygened);
    }

    __host__ AesContext AesContext::ExpandOnHost(uint8_t *key) {
        __m128i roundKeys[aes::NUM_ROUNDS + 1];
        // assert NUM_ROUNDS == 10
        auto keyNum = _mm_loadu_si128(reinterpret_cast<const __m128i*>(key));
        roundKeys[0] = keyNum;
        roundKeys[1] = AESNI128_KEY_EXPAND(roundKeys[0], 1);
        roundKeys[2] = AESNI128_KEY_EXPAND(roundKeys[1], 2);
        roundKeys[3] = AESNI128_KEY_EXPAND(roundKeys[2], 4);
        roundKeys[4] = AESNI128_KEY_EXPAND(roundKeys[3], 8);
        roundKeys[5] = AESNI128_KEY_EXPAND(roundKeys[4], 16);
        roundKeys[6] = AESNI128_KEY_EXPAND(roundKeys[5], 32);
        roundKeys[7] = AESNI128_KEY_EXPAND(roundKeys[6], 64);
        roundKeys[8] = AESNI128_KEY_EXPAND(roundKeys[7], 128);
        roundKeys[9] = AESNI128_KEY_EXPAND(roundKeys[8], 27);
        roundKeys[10] = AESNI128_KEY_EXPAND(roundKeys[9], 54);
        return { reinterpret_cast<aes::u128*>(roundKeys) };
    }

    __device__ aes::u128 AesContext::GenNoise(aes::u128 block) {
        aes::Run(&block, roundKeys);
        return block;
    }
    __device__ aes::u128 AesContext::GenNoiseB(aes::u128 block, aes::u128 *rk) {
        aes::Run(&block, rk);
        return block;
    }
}