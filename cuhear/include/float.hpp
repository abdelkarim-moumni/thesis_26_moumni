#pragma once

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <ieee754.h>
#include <bitset>

#define FLOAT_MANTISSA 21
#define FLOAT_EXPONENT 10
#define IEEE_FLOAT_MANTISSA 23
#define IEEE_FLOAT_EXPONENT 8
#define SHIFT 0 // (FLOAT_EXPONENT - IEEE_FLOAT_EXPONENT)

namespace cuhear::floats {

    union FloatBits
    {
        ieee754_float ieee_float;
        float native_float;

        struct {
            unsigned int mantissa : FLOAT_MANTISSA;
            signed int exponent : FLOAT_EXPONENT;
            unsigned int sign : 1;
        } crypto;

        struct {
            unsigned int remainder : FLOAT_MANTISSA + FLOAT_EXPONENT;
            unsigned int sign : 1;
        } crypto_simplified;
    };
}