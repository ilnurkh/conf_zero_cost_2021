#pragma once
#include "dot_product.h"

template<class TDotProductImpl>
struct TPackedProductUnpack {
    inline static void MultiDotProduct(
        const float* a,
        const uint8_t* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float bias,
        float coeff,
        float* results
    ) {
        std::vector<float> buf(dim);
        for(size_t e = 0; e < elemsNum; e += 1) {
            const uint8_t* rowPtr = allB + dim * elemsIds[e];
            for(size_t i = 0; i < dim; i += 1) {
                buf[i] = coeff * rowPtr[i] + bias;
            }
            results[e] = TDotProductImpl::DotProduct(a, buf.cbegin(), dim);
        }
    }
};

struct TPackedProductInlined {
    inline static void MultiDotProduct(
        const float* a,
        const uint8_t* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float bias,
        float coeff,
        float* results
    ) {
        for(size_t e = 0; e < elemsNum; e += 1) {
            results[e] = 0;
            const uint8_t* rowPtr = allB + dim * elemsIds[e];
            for(size_t i = 0; i < dim; i += 1) {
                results[e] += a[i] * (coeff * rowPtr[i] + bias);
            }
        }
    }
};

struct TPackedProductInlinedWithMath {
    inline static void MultiDotProduct(
        const float* a,
        const uint8_t* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float bias,
        float coeff,
        float* results
    ) {
        float bb = 0;
        for(size_t i = 0; i < dim; i += 1) {
            bb += a[i];
        }
        bb *= bias;

        for(size_t e = 0; e < elemsNum; e += 1) {
            results[e] = 0;
            const uint8_t* rowPtr = allB + dim * elemsIds[e];
            for(size_t i = 0; i < dim; i += 1) {
                results[e] += a[i] * float(rowPtr[i]);
            }
            results[e] = results[e] * coeff + bb;
        }
    }
};

struct TPackedProductInlinedAvx512Auto {
    static void MultiDotProduct(
        const float* a,
        const uint8_t* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float bias,
        float coeff,
        float* results
    );
};

struct TPackedProductInlinedWithMathAvx512Auto {
    static void MultiDotProduct(
        const float* a,
        const uint8_t* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float bias,
        float coeff,
        float* results
    );
};

struct TPackedProductAvx512ASM {
    static void MultiDotProduct(
        const float* a,
        const uint8_t* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float bias,
        float coeff,
        float* results
    );
};

struct TPackedProductV2Avx512ASM {
    static void MultiDotProduct(
        const float* a,
        const uint8_t* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float bias,
        float coeff,
        float* results
    );
};

