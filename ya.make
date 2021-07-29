OWNER(
    alexmir0x1
)

G_BENCHMARK()

TAG(ya:fat)

SRCS(
    stand.cpp
    sse4_impls.cpp
)

SRC_CPP_SSE4(
    sse4_optimizations.cpp -funsafe-math-optimizations
)

SRC_CPP_AVX(
    avx_impls.cpp -funsafe-math-optimizations
)

SRC_CPP_AVX2(
    avx2_impls.cpp -funsafe-math-optimizations
)

SRC_CPP_SSE4(
    avx512_impls.cpp -funsafe-math-optimizations
    -mavx512f -mavx512bw -mavx512cd -mavx512dq -mavx512vl
)

END()
