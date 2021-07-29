#include "dot_product.h"
#include "multidot.h"

float TNaiveOutlined::DotProduct(const float* a, const float* b, size_t dim) {
    return TNaive::DotProduct(a, b, dim);
}

float TCompileTime64hDimOutlined::DotProduct(const float* a, const float* b, size_t dim) {
    return TCompileTimehDim<64>::DotProduct(a, b, dim);
}

float TDetectOptimistic::DotProduct(const float* a, const float* b, size_t dim) {
    if (TRuntimeCpuInfoDispatch::HaveAvx512) {
        return TNaiveAvx512Auto::DotProduct(a, b, dim);
    }
    if (TRuntimeCpuInfoDispatch::HaveAvx2) {
        return TNaiveAvx2Auto::DotProduct(a, b, dim);
    }
    if (TRuntimeCpuInfoDispatch::HaveAvx) {
        return TNaiveAvxAuto::DotProduct(a, b, dim);
    }
    return TBy4SSE4UnsafeOpt::DotProduct(a, b, dim);
}

float TDetectPessimistic::DotProduct(const float* a, const float* b, size_t dim) {
    if (TRuntimeCpuInfoDispatch::HaveSse4Only) {
        return TBy4SSE4UnsafeOpt::DotProduct(a, b, dim);
    }
    if (TRuntimeCpuInfoDispatch::HaveAvxOnly) {
        return TNaiveAvxAuto::DotProduct(a, b, dim);
    }
    if (TRuntimeCpuInfoDispatch::HaveAvx2Only) {
        return TNaiveAvx2Auto::DotProduct(a, b, dim);
    }
    //if (TRuntimeCpuInfoDispatch::HaveAvx512) {
        return TNaiveAvx512Auto::DotProduct(a, b, dim);
    //}
}

float TDetectJump::DotProduct(const float* a, const float* b, size_t dim) {
    switch (TRuntimeCpuInfoDispatch::LevelJump) {
        case 0: return TBy4SSE4UnsafeOpt::DotProduct(a, b, dim);
        case 1: return TNaiveAvxAuto::DotProduct(a, b, dim);
        case 2: return TNaiveAvx2Auto::DotProduct(a, b, dim);
        case 3: return TNaiveAvx512Auto::DotProduct(a, b, dim);
        default: __builtin_unreachable();
    }
}

float TVirtualJump::DotProduct(const float* a, const float* b, size_t dim) {
    return TRuntimeCpuInfoDispatch::Fabric->VDotProduct(a, b, dim);
}

#define DeclByStep(Step) \
template<> void TMultiDotCTStepOutlined<Step>::MultiDotProduct(\
    const float* a,\
    const float* allB,\
    size_t dim,\
    const uint32_t* elemsIds,\
    size_t elemsNum,\
    float* results\
) {\
    TMultiDotCTStep<Step>::MultiDotProduct(a, allB, dim, elemsIds, elemsNum, results);\
}\
template<> void TMultiDotCTStepOutlinedV2<Step>::MultiDotProduct(\
    const float* a,\
    const float* allB,\
    size_t dim,\
    const uint32_t* elemsIds,\
    size_t elemsNum,\
    float* results\
) {\
    TMultiDotCTStepV2<Step>::MultiDotProduct(a, allB, dim, elemsIds, elemsNum, results);\
}

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
