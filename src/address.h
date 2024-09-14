#pragma once
#include "curve_math.h"
#include "keccak.h"
#include "math.h"


__global__ void __launch_bounds__(BLOCK_SIZE) gpu_address_init(CurvePoint* block_offsets, CurvePoint* offsets) {
    bool b = __isGlobal(block_offsets);
    __builtin_assume(b);
    bool b2 = __isGlobal(offsets);
    __builtin_assume(b2);

    uint64_t thread_id = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)BLOCK_SIZE;

    _uint256 z[BLOCK_SIZE];
    z[0] = sub_256_mod_p(block_offsets[thread_id].x, thread_offsets[0].x);

    for (int i = 1; i < BLOCK_SIZE; i++) {
        _uint256 x_delta = sub_256_mod_p(block_offsets[thread_id].x, thread_offsets[i].x);
        z[i] = mul_256_mod_p(z[i - 1], x_delta);
    }

    _uint256 q = eeuclid_256_mod_p(z[BLOCK_SIZE - 1]);

    for (int i = BLOCK_SIZE - 1; i >= 1; i--) {
        _uint256 y = mul_256_mod_p(q, z[i - 1]);
        q = mul_256_mod_p(q, sub_256_mod_p(block_offsets[thread_id].x, thread_offsets[i].x));

        _uint256 lambda = mul_256_mod_p(sub_256_mod_p(block_offsets[thread_id].y, thread_offsets[i].y), y);
        _uint256 curve_x = sub_256_mod_p(sub_256_mod_p(mul_256_mod_p(lambda, lambda), block_offsets[thread_id].x), thread_offsets[i].x);
        _uint256 curve_y = sub_256_mod_p(mul_256_mod_p(lambda, sub_256_mod_p(block_offsets[thread_id].x, curve_x)), block_offsets[thread_id].y);
        offsets[thread_id * BLOCK_SIZE + i] = CurvePoint{curve_x, curve_y};
    }

    _uint256 y = q;
    _uint256 lambda = mul_256_mod_p(sub_256_mod_p(block_offsets[thread_id].y, thread_offsets[0].y), y);
    _uint256 curve_x = sub_256_mod_p(sub_256_mod_p(mul_256_mod_p(lambda, lambda), block_offsets[thread_id].x), thread_offsets[0].x);
    _uint256 curve_y = sub_256_mod_p(mul_256_mod_p(lambda, sub_256_mod_p(block_offsets[thread_id].x, curve_x)), block_offsets[thread_id].y);
    offsets[thread_id * BLOCK_SIZE] = CurvePoint{curve_x, curve_y};
}


__global__ void __launch_bounds__(BLOCK_SIZE, 2) gpu_address_work(CurvePoint* offsets, Address* addresses) {
    // 确认 offsets 是全局内存中的指针，便于编译器优化
    bool b = __isGlobal(offsets);
    __builtin_assume(b);

    // 计算线程 ID 和密钥初始值
    uint64_t thread_id = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)BLOCK_SIZE;
    uint64_t key = (uint64_t)THREAD_WORK * thread_id;

    // 从 offsets 中获取公钥点
    CurvePoint p = offsets[thread_id];

    // 计算第一个地址，并存储到 addresses 数组中
    addresses[key] = calculate_address(p.x, p.y);
    addresses[key + 1] = calculate_address(p.x, sub_256(P, p.y));

    // 椭圆曲线点计算，存储每个中间计算结果
    _uint256 z[THREAD_WORK - 1];
    z[0] = sub_256_mod_p(p.x, addends[0].x);

    for (int i = 1; i < THREAD_WORK - 1; i++) {
        _uint256 x_delta = sub_256_mod_p(p.x, addends[i].x);
        z[i] = mul_256_mod_p(z[i - 1], x_delta);
    }

    _uint256 q = eeuclid_256_mod_p(z[THREAD_WORK - 2]);

    // 从后向前计算曲线点并生成新的地址
    for (int i = THREAD_WORK - 2; i >= 1; i--) {
        _uint256 y = mul_256_mod_p(q, z[i - 1]);
        q = mul_256_mod_p(q, sub_256_mod_p(p.x, addends[i].x));

        _uint256 lambda = mul_256_mod_p(sub_256_mod_p(p.y, addends[i].y), y);
        _uint256 curve_x = sub_256_mod_p(sub_256_mod_p(mul_256_mod_p(lambda, lambda), p.x), addends[i].x);
        _uint256 curve_y = sub_256_mod_p(mul_256_mod_p(lambda, sub_256_mod_p(p.x, curve_x)), p.y);

        // 存储生成的地址
        addresses[key + (i + 1) * 2] = calculate_address(curve_x, curve_y);
        addresses[key + (i + 1) * 2 + 1] = calculate_address(curve_x, sub_256(P, curve_y));
    }

    // 最后一次椭圆曲线运算并存储地址
    _uint256 y = q;
    _uint256 lambda = mul_256_mod_p(sub_256_mod_p(p.y, addends[0].y), y);
    _uint256 curve_x = sub_256_mod_p(sub_256_mod_p(mul_256_mod_p(lambda, lambda), p.x), addends[0].x);
    _uint256 curve_y = sub_256_mod_p(mul_256_mod_p(lambda, sub_256_mod_p(p.x, curve_x)), p.y);

    addresses[key + THREAD_WORK * 2] = calculate_address(curve_x, curve_y);
    addresses[key + THREAD_WORK * 2 + 1] = calculate_address(curve_x, sub_256(P, curve_y));
}
