#include "dot_product.h"
#include "multidot.h"
#include "dotpacked.h"

#include <benchmark/benchmark.h>
#include <vector>
#include <util/random/fast.h>
#include <util/generic/xrange.h>
#include <iostream>

using TRandomGen = TFastRng64;

constexpr size_t MaxDim = 1024;
constexpr size_t MaxRowNumber = 1024u * 1024u;
constexpr size_t MatrixSize = MaxDim * MaxRowNumber;
constexpr size_t CasesNumPerTask = 10 * 1024u;
constexpr size_t TasksNum = 100u;

const bool TRuntimeCpuInfoDispatch::HaveAvx = true;
const bool TRuntimeCpuInfoDispatch::HaveAvx2 = true;
const bool TRuntimeCpuInfoDispatch::HaveAvx512 = true;

const bool TRuntimeCpuInfoDispatch::HaveSse4Only = false;
const bool TRuntimeCpuInfoDispatch::HaveAvxOnly = false;
const bool TRuntimeCpuInfoDispatch::HaveAvx2Only = false;

const uint32_t TRuntimeCpuInfoDispatch::LevelJump = 3;

const std::unique_ptr<const IDotProduct> TRuntimeCpuInfoDispatch::Fabric = std::unique_ptr<const IDotProduct>(
    new IDotProductMaker<TNaiveAvx512Auto>{}
);

// #define B_RANGES Arg(64)
#define B_RANGES Arg(64)->Arg(128)->Arg(1024)
// #define B_RANGES DenseRange(64, 1024, 64)

struct TCalcTask {
    std::vector<float> Query;
    std::vector<ui32> DocIds;

    void Generate(TRandomGen& g) {
        Query.reserve(MaxDim);
        for(size_t i = 0; i < MaxDim; ++i) {
            Query.push_back(g.GenRandReal1() - 0.5f);
        }
        for(unsigned char* b = (unsigned char*)Query.begin(); b != (unsigned char*)Query.end(); b += 1) {
            *b |= 1;
        }

        DocIds.reserve(CasesNumPerTask);
        for(size_t i = 0; i < CasesNumPerTask; ++i) {
            DocIds.push_back(g.Uniform(MaxRowNumber));
        }
    }
};

struct TBaseHolder {
    std::vector<float> Matrix;
    std::vector<uint8_t> Matrix8;
    std::vector<TCalcTask> Tasks;

    void Generate(TRandomGen& g) {
        Matrix.reserve(MatrixSize);
        Matrix8.reserve(MatrixSize);
        for(size_t i = 0; i < MatrixSize; ++i) {
            float x = g.GenRandReal1();
            Matrix.push_back(x - 0.5f);
            Matrix8.push_back(uint8_t(1) + uint8_t(x * 254));
        }
        for(unsigned char* b = (unsigned char*)Matrix.begin(); b != (unsigned char*)Matrix.end(); b += 1) {
            *b |= 1;
        }

        Tasks.resize(TasksNum);
        for(size_t i = 0; i < TasksNum; ++i) {
            Tasks[i].Generate(g);
        }
    }

    TBaseHolder() {
        std::cout << "Start init" << std::endl;
        TRandomGen gen(29);
        Generate(gen);
        std::cout << "Done init" << std::endl;

        #define Check(name) std::cout << \
            name::DotProduct(Tasks[0].Query.cbegin(), Matrix.cbegin(), 64) \
        << "\t" << #name << std::endl;

        #define CheckD(name) std::cout << \
            name::DotProductDouble(Tasks[0].Query.cbegin(), Matrix.cbegin(), Matrix.cbegin() + 64, 64).first \
            << "\t" << \
            name::DotProductDouble(Tasks[0].Query.cbegin(), Matrix.cbegin(), Matrix.cbegin() + 64, 64).second \
        << "\t" << #name << std::endl;


        Check(TNaive);
        Check(TNaiveOutlined);
        Check(TCompileTimehDim<64>);
        Check(TCompileTime64hDimOutlined);
        Check(TCLibChecker)
        Check(TForceBy4)
        Check(TForceBy4Last)
        Check(TNaiveSSE4UnsafeOpt)
        Check(TBy4SSE4UnsafeOpt)
        Check(TNaiveAvxAuto)
        Check(TNaiveAvx2Auto)
        Check(TNaiveAvx512Auto)
        Check(TNaiveAvx512ASM)
        Check(TDetectOptimistic)
        Check(TDetectPessimistic)
        Check(TDetectJump)
        Check(TVirtualJump)
        CheckD(TNaive);

        #define CheckMD(name) {\
            float res[16];\
            uint32_t elems[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};\
            name::MultiDotProduct(Tasks[0].Query.cbegin(), Matrix.cbegin(), 64, elems, 16, res);\
            std::cout << res[0] << "\t" << res[1] << "\t" << #name << std::endl;\
        }

        CheckMD(TMultiDotFromSingle<TNaive>);
        CheckMD(TMultiDotAll);
        CheckMD(TMultiDotCTStep<1>);
        CheckMD(TMultiDotCTStep<2>);
        CheckMD(TMultiDotCTStepOutlined<2>);
        CheckMD(TMultiDotCTStepOutlinedV2<2>);
        CheckMD(TMultiDotCTStepV3);
        CheckMD(TMultiDotCTStepV2FloatOpts_SSE42<2>);
        CheckMD(TMultiDotCTStepV2FloatOpts_AVX<2>);
        CheckMD(TMultiDotCTStepV2FloatOpts_AVX2<2>);
        CheckMD(TMultiDotCTStepV2FloatOpts_AVX512<2>);
        CheckMD(TMultiDotV3_ASM_AVX512);

        #define CheckPacked(name) {\
            float res[16];\
            uint32_t elems[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};\
            name::MultiDotProduct(Tasks[0].Query.cbegin(), Matrix8.cbegin(), 64, elems, 16, 0.7, 0.5, res);\
            std::cout << res[0] << "\t" << res[1] << "\t" << #name << std::endl;\
        }

        CheckPacked(TPackedProductUnpack<TNaive>);
        CheckPacked(TPackedProductUnpack<TVirtualJump>);
        CheckPacked(TPackedProductInlined);
        CheckPacked(TPackedProductInlinedAvx512Auto);
        CheckPacked(TPackedProductInlinedWithMath);
        CheckPacked(TPackedProductInlinedWithMathAvx512Auto);
        CheckPacked(TPackedProductAvx512ASM);
        CheckPacked(TPackedProductV2Avx512ASM);
    }
} Base;

template<class TProductImpl>
inline void DotProductBench(benchmark::State& state) {
    size_t taskId = 0;
    size_t dim = state.range(0);
    for (auto _ : state) {
        for(size_t c = 0; c < CasesNumPerTask; c += 1) {
            size_t rowId = Base.Tasks[taskId].DocIds[c];
            float caseRes = TProductImpl::DotProduct(
                Base.Tasks[taskId].Query.cbegin(),
                Base.Matrix.cbegin() + rowId * dim,
                dim
            );
            benchmark::DoNotOptimize(caseRes);
        }
        taskId += 1;
        taskId = taskId % TasksNum;
    }
}

#define DeclareBenchN(CL, name) \
static void DotPr_##name(benchmark::State& state) {DotProductBench<CL>(state);} \
BENCHMARK(DotPr_##name)->Unit(benchmark::kMillisecond)

#define DeclareBench(CL) DeclareBenchN(CL, CL)

template<class TProductImpl>
inline void DotProductBenchDouble(benchmark::State& state) {
    size_t taskId = 0;
    size_t dim = state.range(0);
    for (auto _ : state) {
        for(size_t c = 0; c < CasesNumPerTask; c += 2) {
            size_t rowId1 = Base.Tasks[taskId].DocIds[c];
            size_t rowId2 = Base.Tasks[taskId].DocIds[c+1];
            std::pair<float, float> caseRes = TProductImpl::DotProductDouble(
                Base.Tasks[taskId].Query.cbegin(),
                Base.Matrix.cbegin() + rowId1 * dim,
                Base.Matrix.cbegin() + rowId2 * dim,
                dim
            );
            benchmark::DoNotOptimize(caseRes);
        }
        taskId += 1;
        taskId = taskId % TasksNum;
    }
}


#define DeclareBenchDoubleN(CL, name) \
static void DotPrD_##name(benchmark::State& state) {DotProductBenchDouble<CL>(state);} \
BENCHMARK(DotPrD_##name)->Unit(benchmark::kMillisecond)

#define DeclareBenchDouble(CL) DeclareBenchDoubleN(CL, CL)

template<class TProductImpl>
inline void DotProductBenchMulti(benchmark::State& state) {
    size_t taskId = 0;
    size_t dim = state.range(0);
    std::vector<float> results(CasesNumPerTask, 0.f);
    for (auto _ : state) {
        TProductImpl::MultiDotProduct(
            Base.Tasks[taskId].Query.cbegin(),
            Base.Matrix.cbegin(),
            dim,
            Base.Tasks[taskId].DocIds.cbegin(),
            Base.Tasks[taskId].DocIds.size(),
            results.begin()
        );
        benchmark::DoNotOptimize(results);
        taskId += 1;
        taskId = taskId % TasksNum;
    }
}

#define DeclareBenchMultiN(CL, name) \
static void DotPrMulti_##name(benchmark::State& state) {DotProductBenchMulti<CL>(state);} \
BENCHMARK(DotPrMulti_##name)->Unit(benchmark::kMillisecond)

#define DeclareBenchMulti(CL) DeclareBenchMultiN(CL, CL)


DeclareBenchN(TCompileTimehDim<64>, TCompileTimehDim64)
    ->DenseRange(64, 64, 64);

DeclareBench(TCompileTime64hDimOutlined)
    ->DenseRange(64, 64, 64);

DeclareBench(TCLibChecker)
    ->B_RANGES;

DeclareBench(TNaive)
    ->B_RANGES;

DeclareBenchDouble(TNaive)
    ->B_RANGES;

DeclareBench(TNaiveOutlined)
    ->B_RANGES;

DeclareBench(TNaiveSSE4UnsafeOpt)
    ->B_RANGES;

DeclareBench(TForceBy4)
    ->B_RANGES;

DeclareBench(TForceBy4Last)
    ->B_RANGES;

DeclareBench(TBy4SSE4UnsafeOpt)
    ->B_RANGES;

DeclareBench(TNaiveAvxAuto)
    ->B_RANGES;
DeclareBench(TNaiveAvx2Auto)
    ->B_RANGES;
DeclareBench(TNaiveAvx512Auto)
    ->B_RANGES;
DeclareBench(TNaiveAvx512ASM)
    ->B_RANGES;

DeclareBench(TDetectOptimistic)
    ->B_RANGES;
DeclareBench(TDetectPessimistic)
    ->B_RANGES;
DeclareBench(TDetectJump)
    ->B_RANGES;
DeclareBench(TVirtualJump)
    ->B_RANGES;

DeclareBenchMulti(TMultiDotAll)
    ->B_RANGES;
DeclareBenchMultiN(TMultiDotFromSingle<TNaive>, TMultiDotFromSingle_Naive)
    ->B_RANGES;
DeclareBenchMultiN(TMultiDotFromSingle<TNaiveOutlined>, TMultiDotFromSingle_NaiveOutlined)
    ->B_RANGES;

DeclareBenchMultiN(TMultiDotCTStep<1>, TMultiDotCTStep_1)
    ->B_RANGES;

DeclareBenchMulti(TMultiDotCTStepV3)
    ->B_RANGES;
DeclareBenchMulti(TMultiDotCTStepV3FloatOpts_SSE42)
    ->B_RANGES;
DeclareBenchMulti(TMultiDotCTStepV3FloatOpts_AVX)
    ->B_RANGES;
DeclareBenchMulti(TMultiDotCTStepV3FloatOpts_AVX2)
    ->B_RANGES;
DeclareBenchMulti(TMultiDotCTStepV3FloatOpts_AVX512)
    ->B_RANGES;
DeclareBenchMulti(TMultiDotV3_ASM_PREFETCH_AVX512)
    ->B_RANGES;
DeclareBenchMulti(TMultiDotV3_ASM_AVX512)
    ->B_RANGES;

#define DeclareMultiDotVariantsByStep(Step)\
DeclareBenchMultiN(TMultiDotCTStep<Step>, TMultiDotCTStep_##Step)\
    ->B_RANGES;\
DeclareBenchMultiN(TMultiDotCTStepOutlined<Step>, TMultiDotCTStepOutlined_##Step)\
    ->B_RANGES;\
DeclareBenchMultiN(TMultiDotCTStepOutlinedV2<Step>, TMultiDotCTStepOutlinedV2_##Step)\
    ->B_RANGES;\
DeclareBenchMultiN(TMultiDotCTStepV2FloatOpts_SSE42<Step>, TMultiDotCTStepV2FloatOpts_SSE42_##Step)\
    ->B_RANGES;\
DeclareBenchMultiN(TMultiDotCTStepV2FloatOpts_AVX<Step>, TMultiDotCTStepV2FloatOpts_AVX_##Step)\
    ->B_RANGES;\
DeclareBenchMultiN(TMultiDotCTStepV2FloatOpts_AVX2<Step>, TMultiDotCTStepV2FloatOpts_AVX2_##Step)\
    ->B_RANGES;\
DeclareBenchMultiN(TMultiDotCTStepV2FloatOpts_AVX512<Step>, TMultiDotCTStepV2FloatOpts_AVX512_##Step)\
    ->B_RANGES;\

DeclareMultiDotVariantsByStep(2)
DeclareMultiDotVariantsByStep(3)
DeclareMultiDotVariantsByStep(4)
DeclareMultiDotVariantsByStep(5)
DeclareMultiDotVariantsByStep(6)
// DeclareMultiDotVariantsByStep(7)
// DeclareMultiDotVariantsByStep(8)
// DeclareMultiDotVariantsByStep(9)
// DeclareMultiDotVariantsByStep(10)
// DeclareMultiDotVariantsByStep(11)
// DeclareMultiDotVariantsByStep(12)
// DeclareMultiDotVariantsByStep(13)
// DeclareMultiDotVariantsByStep(14)
// DeclareMultiDotVariantsByStep(15)
// DeclareMultiDotVariantsByStep(16)


template<class TProductImpl>
inline void PackedDotProductBenchMulti(benchmark::State& state) {
    size_t taskId = 0;
    size_t dim = state.range(0);
    std::vector<float> results(CasesNumPerTask, 0.f);
    for (auto _ : state) {
        TProductImpl::MultiDotProduct(
            Base.Tasks[taskId].Query.cbegin(),
            Base.Matrix8.cbegin(),
            dim,
            Base.Tasks[taskId].DocIds.cbegin(),
            Base.Tasks[taskId].DocIds.size(),
            0.7,
            0.4,
            results.begin()
        );
        benchmark::DoNotOptimize(results);
        taskId += 1;
        taskId = taskId % TasksNum;
    }
}

#define DeclareBenchMultiPackedN(CL, name) \
static void DotPrMultiPacked_##name(benchmark::State& state) {PackedDotProductBenchMulti<CL>(state);} \
BENCHMARK(DotPrMultiPacked_##name)->Unit(benchmark::kMillisecond)

#define DeclareBenchMultiPacked(CL) DeclareBenchMultiPackedN(CL, CL)

DeclareBenchMultiPackedN(TPackedProductUnpack<TNaive>, Packed_Naive)
    ->B_RANGES;

DeclareBenchMultiPackedN(TPackedProductUnpack<TVirtualJump>, Packed_Virtual)
    ->B_RANGES;

DeclareBenchMultiPacked(TPackedProductInlined)
    ->B_RANGES;
DeclareBenchMultiPacked(TPackedProductInlinedAvx512Auto)
    ->B_RANGES;

DeclareBenchMultiPacked(TPackedProductInlinedWithMath)
    ->B_RANGES;
DeclareBenchMultiPacked(TPackedProductInlinedWithMathAvx512Auto)
    ->B_RANGES;
DeclareBenchMultiPacked(TPackedProductAvx512ASM)
    ->B_RANGES;
DeclareBenchMultiPacked(TPackedProductV2Avx512ASM)
    ->B_RANGES;
