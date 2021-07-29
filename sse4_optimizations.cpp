#include "dot_product.h"
#include "multidot.h"

float TNaiveSSE4UnsafeOpt::DotProduct(const float* a, const float* b, size_t dim) {
    return TNaive::DotProduct(a, b, dim);
}

float TBy4SSE4UnsafeOpt::DotProduct(const float* a, const float* b, size_t dim) {
    return TForceBy4::DotProduct(a, b, dim);
}

#define DeclByStep(Step) \
template<> void TMultiDotCTStepV2FloatOpts_SSE42<Step>::MultiDotProduct(\
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

void TMultiDotCTStepV3FloatOpts_SSE42::MultiDotProduct(
    const float* a,
    const float* allB,
    size_t dim,
    const uint32_t* elemsIds,
    size_t elemsNum,
    float* results
) {
    TMultiDotCTStepV3::MultiDotProduct(a, allB, dim, elemsIds, elemsNum, results);
}
