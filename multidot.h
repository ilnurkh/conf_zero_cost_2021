#pragma once

#include "dot_product.h"


template<class Basic>
struct TMultiDotFromSingle {
    inline static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    ) {
        for(size_t e = 0; e < elemsNum; e += 1) {
            results[e] = Basic::DotProduct(a, allB + dim * elemsIds[e], dim);
        }
    }
};

struct TMultiDotAll {
    inline static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    ) {
        for(size_t e = 0; e < elemsNum; e += 1) {
            results[e] = 0;
        }
        for(size_t i = 0; i < dim; i += 1) {
            for(size_t e = 0; e < elemsNum; e += 1) {
                results[e] += a[i] * *(allB + dim * elemsIds[e] + i);
            }
        }
    }
};

template<size_t Step>
struct TMultiDotCTStep {
    inline static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    ) {
        size_t e = 0;
        for(; e + Step <= elemsNum; e += Step) {
            for(size_t ee = 0; ee < Step; ee += 1) { //unroll by constexpr size
                results[e + ee] = 0;
            }
            for(size_t i = 0; i < dim; i += 1) {
                for(size_t ee = 0; ee < Step; ee += 1) { //unroll by constexpr size
                    results[e + ee] += a[i] * *(allB + dim * elemsIds[e + ee] + i);
                }
            }
        }
        for(; e < elemsNum; e += 1) {
            results[e] = TNaiveSSE4UnsafeOpt::DotProduct(a, allB + dim * elemsIds[e], dim);
        }
    }
};

template<size_t Step>
struct TMultiDotCTStepV2 {
    inline static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    ) {
        size_t e = 0;
        for(; e + Step <= elemsNum; e += Step) {
            float tmp[Step];
            for(size_t ee = 0; ee < Step; ee += 1) { //unroll by constexpr size
                tmp[ee] = 0;
            }
            for(size_t i = 0; i < dim; i += 1) {
                for(size_t ee = 0; ee < Step; ee += 1) { //unroll by constexpr size
                    tmp[ee] += a[i] * *(allB + dim * elemsIds[e + ee] + i);
                }
            }
            for(size_t ee = 0; ee < Step; ee += 1) { //unroll by constexpr size
                results[e + ee] = tmp[ee];
            }
        }
        for(; e < elemsNum; e += 1) {
            results[e] = TNaiveSSE4UnsafeOpt::DotProduct(a, allB + dim * elemsIds[e], dim);
        }
    }
};

struct TMultiDotCTStepV3 {
    inline static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    ) {
        size_t constexpr Step = 4;
        size_t e = 0;
        float tmp0 = 0;
        float tmp1 = 0;
        float tmp2 = 0;
        float tmp3 = 0;
        // float tmp4 = 0;
        // float tmp5 = 0;
        // float tmp6 = 0;
        // float tmp7 = 0;
        // float tmp8 = 0;
        for(; e + Step <= elemsNum; e += Step) {
            const float* r0 = allB + dim * elemsIds[e + 0];
            const float* r1 = allB + dim * elemsIds[e + 1];
            const float* r2 = allB + dim * elemsIds[e + 2];
            const float* r3 = allB + dim * elemsIds[e + 3];
            // const float* r4 = allB + dim * elemsIds[e + 4];
            // const float* r5 = allB + dim * elemsIds[e + 5];
            // const float* r6 = allB + dim * elemsIds[e + 6];
            // const float* r7 = allB + dim * elemsIds[e + 7];

            for(size_t i = 0; i < dim; i += 1) {
                tmp0 += a[i] * r0[i];
                tmp1 += a[i] * r1[i];
                tmp2 += a[i] * r2[i];
                tmp3 += a[i] * r3[i];
                // tmp4 += a[i] * r4[i];
                // tmp5 += a[i] * r5[i];
                // tmp6 += a[i] * r6[i];
                // tmp7 += a[i] * r7[i];
            }

            results[e + 0] = tmp0;
            results[e + 1] = tmp1;
            results[e + 2] = tmp2;
            results[e + 3] = tmp3;
            // results[e + 4] = tmp4;
            // results[e + 5] = tmp5;
            // results[e + 6] = tmp6;
            // results[e + 7] = tmp7;
        }
        for(; e < elemsNum; e += 1) {
            results[e] = TNaiveSSE4UnsafeOpt::DotProduct(a, allB + dim * elemsIds[e], dim);
        }
    }
};

template<size_t Step>
struct TMultiDotCTStepOutlined {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};

template<size_t Step>
struct TMultiDotCTStepOutlinedV2 {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};

template<size_t Step>
struct TMultiDotCTStepV2FloatOpts_SSE42 {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};

struct TMultiDotCTStepV3FloatOpts_SSE42 {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};

template<size_t Step>
struct TMultiDotCTStepV2FloatOpts_AVX {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};

struct TMultiDotCTStepV3FloatOpts_AVX {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};

template<size_t Step>
struct TMultiDotCTStepV2FloatOpts_AVX2 {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};

struct TMultiDotCTStepV3FloatOpts_AVX2 {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};

template<size_t Step>
struct TMultiDotCTStepV2FloatOpts_AVX512 {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};

struct TMultiDotCTStepV3FloatOpts_AVX512 {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};

struct TMultiDotV3_ASM_AVX512 {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};

struct TMultiDotV3_ASM_PREFETCH_AVX512 {
    static void MultiDotProduct(
        const float* a,
        const float* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float* results
    );
};
