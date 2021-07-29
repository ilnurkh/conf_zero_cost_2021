#pragma once
#include <vector>
#include <cstring>
#include <cstdint>

struct IDotProduct {
    virtual float VDotProduct(const float* a, const float* b, size_t dim) const  = 0;
    virtual ~IDotProduct() {}
};

template<class T>
struct IDotProductMaker : public IDotProduct {
    float VDotProduct(const float* a, const float* b, size_t dim) const final {
        return T::DotProduct(a, b, dim);
    }
};

struct TNaive {
    inline static float DotProduct(const float* a, const float* b, size_t dim) {
        float res = 0;
        for(size_t i = 0; i < dim; ++i) {
            res += a[i] * b[i];
        }
        return res;
    }

    inline static std::pair<float, float> DotProductDouble(const float* a, const float* b, const float* c, size_t dim) {
        float res1 = 0;
        float res2 = 0;
        for(size_t i = 0; i < dim; ++i) {
            res1 += a[i] * b[i];
            res2 += a[i] * c[i];
        }
        return {res1, res2};
    }
};

struct TForceBy4 {
    inline static float DotProduct(const float* a, const float* b, size_t dim) {
        float res1 = 0;
        float res2 = 0;
        float res3 = 0;
        float res4 = 0;
        for(size_t i = 0; i < dim; i += 4) {
            res1 += a[i] * b[i];
            res2 += a[i + 1] * b[i + 1];
            res3 += a[i + 2] * b[i + 2];
            res4 += a[i + 3] * b[i + 3];
        }
        return res1 + res2 + res3 + res4;
    }
};

struct TForceBy4Last {
    inline static float DotProduct(const float* a, const float* b, size_t dim) {
        float res1 = 0;
        float res2 = 0;
        float res3 = 0;
        float res4 = 0;
        for(size_t i = 0; i < dim; i += 4) {
            res1 += a[i] * b[i];
            res2 += a[i + 1] * b[i + 1];
            res3 += a[i + 2] * b[i + 2];
            res4 += a[i + 3] * b[i + 3];
        }
        return (res1 + res3) + (res2 + res4);
    }
};


struct TNaiveOutlined {
    static float DotProduct(const float* a, const float* b, size_t dim);
};

template <size_t dim>
struct TCompileTimehDim {
    inline static float DotProduct(const float* a, const float* b, size_t) {
        float res = 0;
        for(size_t i = 0; i < dim; ++i) {
            res += a[i] * b[i];
        }
        return res;
    }
};

struct TCompileTime64hDimOutlined {
    float static DotProduct(const float* a, const float* b, size_t dim);
};

struct TCLibChecker {
    inline static float DotProduct(const float* a, const float* b, size_t dim) {
        return
            int((const char*) (memchr(a, 0, dim) ?: a) - (const char*)a)
             + int((const char*) (memchr(b, 0, dim) ?: b) - (const char*)b);
    }
};

struct TNaiveSSE4UnsafeOpt {
    static float DotProduct(const float* a, const float* b, size_t dim);
};

struct TBy4SSE4UnsafeOpt {
    static float DotProduct(const float* a, const float* b, size_t dim);
};

struct TNaiveAvxAuto {
    static float DotProduct(const float* a, const float* b, size_t dim);
};

struct TNaiveAvx2Auto {
    static float DotProduct(const float* a, const float* b, size_t dim);
};

struct TNaiveAvx512Auto {
    static float DotProduct(const float* a, const float* b, size_t dim);
};

struct TNaiveAvx512ASM {
    static float DotProduct(const float* a, const float* b, size_t dim);
};

struct TRuntimeCpuInfoDispatch {
    static const bool HaveAvx;
    static const bool HaveAvx2;
    static const bool HaveAvx512;

    static const bool HaveSse4Only;
    static const bool HaveAvxOnly;
    static const bool HaveAvx2Only;

    static const uint32_t LevelJump; // HaveAvx + HaveAvx2 + HaveAvx512
    static const std::unique_ptr<const IDotProduct> Fabric;
};

struct TDetectOptimistic {
    static float DotProduct(const float* a, const float* b, size_t dim);
};

struct TDetectPessimistic {
    static float DotProduct(const float* a, const float* b, size_t dim);
};

struct TDetectJump {
    static float DotProduct(const float* a, const float* b, size_t dim);
};

struct TVirtualJump {
    static float DotProduct(const float* a, const float* b, size_t dim);
};
