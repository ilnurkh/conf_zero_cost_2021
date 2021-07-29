#include "dot_product.h"
#include "multidot.h"
#include "dotpacked.h"

#include <immintrin.h>

float TNaiveAvx512Auto::DotProduct(const float* a, const float* b, size_t dim) {
    return TNaive::DotProduct(a, b, dim);
}


float TNaiveAvx512ASM::DotProduct(
    const float* a, const float* b, size_t dim
) {
    __m512 sum0 = _mm512_setzero_ps();

    for(size_t position = 0; position < dim; position += (sizeof(__m512) / sizeof(float))) {
        __m512 left = _mm512_load_ps(a + position);
        sum0 = _mm512_fmadd_ps(left, _mm512_load_ps(b + position), sum0);
    }

    return _mm512_reduce_add_ps(sum0);
}

#define DeclByStep(Step) \
template<> void TMultiDotCTStepV2FloatOpts_AVX512<Step>::MultiDotProduct(\
    const float* a,\
    const float* allB,\
    size_t dim,\
    const uint32_t* elemsIds,\
    size_t elemsNum,\
    float* results\
) {\
    TMultiDotCTStepV2<Step>::MultiDotProduct(a, allB, dim, elemsIds, elemsNum, results);\
}\

DeclByStep(2)
DeclByStep(3)
DeclByStep(4)
DeclByStep(5)
DeclByStep(6)
DeclByStep(7)
DeclByStep(8)
DeclByStep(9)
DeclByStep(10)
DeclByStep(11)
DeclByStep(12)
DeclByStep(13)
DeclByStep(14)
DeclByStep(15)
DeclByStep(16)

void TMultiDotCTStepV3FloatOpts_AVX512::MultiDotProduct(
    const float* a,
    const float* allB,
    size_t dim,
    const uint32_t* elemsIds,
    size_t elemsNum,
    float* results
) {
    TMultiDotCTStepV3::MultiDotProduct(a, allB, dim, elemsIds, elemsNum, results);
}

void TMultiDotV3_ASM_AVX512::MultiDotProduct(
    const float* a,
    const float* allB,
    size_t dim,
    const uint32_t* elemsIds,
    size_t elemsNum,
    float* results
) {
    size_t e = 0;
    constexpr size_t Step = 4;
    for(; e + Step <= elemsNum; e += Step) {
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();
        // __m512 sum4 = _mm512_setzero_ps();
        // __m512 sum5 = _mm512_setzero_ps();
        // __m512 sum6 = _mm512_setzero_ps();
        // __m512 sum7 = _mm512_setzero_ps();

        const float* e0 = allB + dim * elemsIds[e + 0];
        const float* e1 = allB + dim * elemsIds[e + 1];
        const float* e2 = allB + dim * elemsIds[e + 2];
        const float* e3 = allB + dim * elemsIds[e + 3];
        // const float* e4 = allB + dim * elemsIds[e + 4];
        // const float* e5 = allB + dim * elemsIds[e + 5];
        // const float* e6 = allB + dim * elemsIds[e + 6];
        // const float* e7 = allB + dim * elemsIds[e + 7];

        for(size_t position = 0; position < dim; position += (sizeof(__m512) / sizeof(float))) {
            __m512 left = _mm512_load_ps(a + position);
            sum0 = _mm512_fmadd_ps(left, _mm512_load_ps(e0 + position), sum0);
            sum1 = _mm512_fmadd_ps(left, _mm512_load_ps(e1 + position), sum1);
            sum2 = _mm512_fmadd_ps(left, _mm512_load_ps(e2 + position), sum2);
            sum3 = _mm512_fmadd_ps(left, _mm512_load_ps(e3 + position), sum3);
            // sum4 = _mm512_fmadd_ps(left, _mm512_load_ps(e4 + position), sum4);
            // sum5 = _mm512_fmadd_ps(left, _mm512_load_ps(e5 + position), sum5);
            // sum6 = _mm512_fmadd_ps(left, _mm512_load_ps(e6 + position), sum6);
            // sum7 = _mm512_fmadd_ps(left, _mm512_load_ps(e7 + position), sum7);
        }

        results[e + 0] = _mm512_reduce_add_ps(sum0);
        results[e + 1] = _mm512_reduce_add_ps(sum1);
        results[e + 2] = _mm512_reduce_add_ps(sum2);
        results[e + 3] = _mm512_reduce_add_ps(sum3);
        // results[e + 4] = _mm512_reduce_add_ps(sum4);
        // results[e + 5] = _mm512_reduce_add_ps(sum5);
        // results[e + 6] = _mm512_reduce_add_ps(sum6);
        // results[e + 7] = _mm512_reduce_add_ps(sum7);
    }
    for(; e < elemsNum; e += 1) {
        results[e] = TNaiveAvx512Auto::DotProduct(a, allB + dim * elemsIds[e], dim);
    }
}

void TMultiDotV3_ASM_PREFETCH_AVX512::MultiDotProduct(
    const float* a,
    const float* allB,
    size_t dim,
    const uint32_t* elemsIds,
    size_t elemsNum,
    float* results
) {
    size_t e = 0;
    constexpr size_t Step = 4;
    constexpr size_t ElemsInVec = (sizeof(__m512) / sizeof(float));
    for(; e + Step <= elemsNum; e += Step) {
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();

        const float* e0 = allB + dim * elemsIds[e + 0];
        const float* e1 = allB + dim * elemsIds[e + 1];
        const float* e2 = allB + dim * elemsIds[e + 2];
        const float* e3 = allB + dim * elemsIds[e + 3];

        for(size_t position = 0; position < dim; position += ElemsInVec) {
            __m512 left = _mm512_load_ps(a + position);
            sum0 = _mm512_fmadd_ps(left, _mm512_load_ps(e0 + position), sum0);
            sum1 = _mm512_fmadd_ps(left, _mm512_load_ps(e1 + position), sum1);
            sum2 = _mm512_fmadd_ps(left, _mm512_load_ps(e2 + position), sum2);
            sum3 = _mm512_fmadd_ps(left, _mm512_load_ps(e3 + position), sum3);
        }
        __builtin_prefetch(a, 0);
        __builtin_prefetch(allB + dim * elemsIds[e + 4 + 0], 0);
        __builtin_prefetch(allB + dim * elemsIds[e + 4 + 1], 0);
        __builtin_prefetch(allB + dim * elemsIds[e + 4 + 2], 0);
        __builtin_prefetch(allB + dim * elemsIds[e + 4 + 3], 0);

        results[e + 0] = _mm512_reduce_add_ps(sum0);
        results[e + 1] = _mm512_reduce_add_ps(sum1);
        results[e + 2] = _mm512_reduce_add_ps(sum2);
        results[e + 3] = _mm512_reduce_add_ps(sum3);
    }
    for(; e < elemsNum; e += 1) {
        results[e] = TNaiveAvx512Auto::DotProduct(a, allB + dim * elemsIds[e], dim);
    }
}



void TPackedProductInlinedAvx512Auto::MultiDotProduct(
        const float* a,
        const uint8_t* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float bias,
        float coeff,
        float* results
    ) {
        TPackedProductInlined::MultiDotProduct(
            a, allB, dim, elemsIds, elemsNum, bias, coeff, results
        );
    }


void TPackedProductInlinedWithMathAvx512Auto::MultiDotProduct(
        const float* a,
        const uint8_t* allB,
        size_t dim,
        const uint32_t* elemsIds,
        size_t elemsNum,
        float bias,
        float coeff,
        float* results
    ) {
        TPackedProductInlinedWithMath::MultiDotProduct(
            a, allB, dim, elemsIds, elemsNum, bias, coeff, results
        );
    }

void TPackedProductAvx512ASM::MultiDotProduct(
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

    size_t e = 0;
    constexpr size_t Step = 4;
    constexpr size_t ElemsInVec = (sizeof(__m512) / sizeof(float));
    for(; e + Step <= elemsNum; e += Step) {
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();

        const uint8_t* e0 = allB + dim * elemsIds[e + 0];
        const uint8_t* e1 = allB + dim * elemsIds[e + 1];
        const uint8_t* e2 = allB + dim * elemsIds[e + 2];
        const uint8_t* e3 = allB + dim * elemsIds[e + 3];

        for(size_t position = 0; position < dim; position += ElemsInVec) {
            __m512 left = _mm512_load_ps(a + position);
            __m512 right0 = _mm512_cvtepi32_ps(
                _mm512_cvtepu16_epi32(
                    _mm256_cvtepu8_epi16(
                        _mm_loadu_epi8(e0 + position)
                    )
                )
            );
            __m512 right1 = _mm512_cvtepi32_ps(
                _mm512_cvtepu16_epi32(
                    _mm256_cvtepu8_epi16(
                        _mm_loadu_epi8(e1 + position)
                    )
                )
            );
            __m512 right2 = _mm512_cvtepi32_ps(
                _mm512_cvtepu16_epi32(
                    _mm256_cvtepu8_epi16(
                        _mm_loadu_epi8(e2 + position)
                    )
                )
            );
            __m512 right3 = _mm512_cvtepi32_ps(
                _mm512_cvtepu16_epi32(
                    _mm256_cvtepu8_epi16(
                        _mm_loadu_epi8(e3 + position)
                    )
                )
            );

            sum0 = _mm512_fmadd_ps(left, right0, sum0);
            sum1 = _mm512_fmadd_ps(left, right1, sum1);
            sum2 = _mm512_fmadd_ps(left, right2, sum2);
            sum3 = _mm512_fmadd_ps(left, right3, sum3);
        }

        results[e + 0] = _mm512_reduce_add_ps(sum0) * coeff + bb;
        results[e + 1] = _mm512_reduce_add_ps(sum1) * coeff + bb;
        results[e + 2] = _mm512_reduce_add_ps(sum2) * coeff + bb;
        results[e + 3] = _mm512_reduce_add_ps(sum3) * coeff + bb;
    }

    TPackedProductInlinedWithMath::MultiDotProduct(a, allB, dim, elemsIds + e, elemsNum - e, bias, coeff, results + e);
}

void TPackedProductV2Avx512ASM::MultiDotProduct(
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

    size_t e = 0;
    constexpr size_t ElemsInVec = (sizeof(__m512) / sizeof(uint8_t));
    constexpr size_t ElemsInVecLeft = (sizeof(__m512) / sizeof(float));
    for(; e < elemsNum; e += 1) {
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();

        const uint8_t* row = allB + dim * elemsIds[e + 0];

        for(size_t position = 0; position < dim; position += ElemsInVec) {
            __m512 left0 = _mm512_load_ps(a + position + 0 * ElemsInVecLeft);
            __m512 left1 = _mm512_load_ps(a + position + 1 * ElemsInVecLeft);
            __m512 left2 = _mm512_load_ps(a + position + 2 * ElemsInVecLeft);
            __m512 left3 = _mm512_load_ps(a + position + 3 * ElemsInVecLeft);

            __m512i rightLoaded = _mm512_loadu_epi8(row + position);
            __m128i right0 = _mm512_extracti32x4_epi32(rightLoaded, 0);
            __m128i right1 = _mm512_extracti32x4_epi32(rightLoaded, 1);
            __m128i right2 = _mm512_extracti32x4_epi32(rightLoaded, 2);
            __m128i right3 = _mm512_extracti32x4_epi32(rightLoaded, 3);

            sum0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(
                _mm512_cvtepu16_epi32(
                    _mm256_cvtepu8_epi16(
                        right0
                    )
                )
            ), left0, sum0);

            sum1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(
                _mm512_cvtepu16_epi32(
                    _mm256_cvtepu8_epi16(
                        right1
                    )
                )
            ), left1, sum1);

            sum2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(
                _mm512_cvtepu16_epi32(
                    _mm256_cvtepu8_epi16(
                        right2
                    )
                )
            ), left2, sum2);

            sum3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(
                _mm512_cvtepu16_epi32(
                    _mm256_cvtepu8_epi16(
                        right3
                    )
                )
            ), left3, sum3);
        }

        results[e] = _mm512_reduce_add_ps(_mm512_add_ps(
            _mm512_add_ps(sum0, sum1),
            _mm512_add_ps(sum2, sum3)
        )) * coeff + bb;
    }
}
