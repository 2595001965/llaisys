#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <algorithm> // std::min
#include <vector>    // std::vector（float 解码缓冲区）

namespace llaisys::ops {

// 计算 Y = X * W^T + b
//   out    : [m, n]，输出 Y，2D 连续张量
//   in     : [m, k]，输入 X，2D 连续张量
//   weight : [n, k]，权重 W，2D 连续张量（注意没有转置，需要在计算中处理）
//   bias   : [n]，偏置 b，1D 张量，可选（不提供时为空指针）
// 全部实现都写在本函数内：m、n、k 等维度都直接从张量对象上读取，不通过参数传入。
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // ---- 参数检查 ----
    // 输出、输入、权重必须在同一设备上。
    CHECK_SAME_DEVICE(out, in, weight);
    // 三者数据类型必须一致。
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    // 维度约束：三者都是二维。
    ASSERT(out->ndim() == 2, "Linear: out must be a 2D tensor.");
    ASSERT(in->ndim() == 2, "Linear: in must be a 2D tensor.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be a 2D tensor.");
    // 暂时只支持连续张量。
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: all tensors must be contiguous.");

    // 从张量形状取出三个维度：m 为行数，k 为输入特征数，n 为输出特征数（weight 的行数）。
    const size_t m = in->shape()[0];
    const size_t k = in->shape()[1];
    const size_t n = weight->shape()[0];

    // 权重形状必须是 [out_features, in_features]，输出形状必须是 [m, n]。
    ASSERT(weight->shape()[1] == k, "Linear: weight shape must be [out_features, in_features].");
    ASSERT(out->shape()[0] == m && out->shape()[1] == n,
           "Linear: out shape must be [in.shape[0], weight.shape[0]].");

    // bias 是可选的：tensor_t 是 shared_ptr，空指针即表示“不提供偏置”。
    // 必须先判空再访问，否则解引用空 shared_ptr 属于未定义行为。
    const bool has_bias = static_cast<bool>(bias);
    if (has_bias) {
        CHECK_SAME_DEVICE(out, bias);                  // 偏置需与输出同设备
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype()); // 偏置需与输出同类型
        ASSERT(bias->ndim() == 1, "Linear: bias must be a 1D tensor.");
        ASSERT(bias->shape()[0] == n, "Linear: bias length must match out_features.");
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    // ---- 设备分派：CPU 始终可用，其余设备暂不支持 ----
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        // 先把线程上下文切换到目标设备。
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
        if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
            // CUDA 版本留待后续作业实现。
            TO_BE_IMPLEMENTED();
            return;
        }
#endif
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    // ---- 以下是 CPU 实现 ----
    // 第 1 步：把输入、权重、偏置统一整理成 float 数组。
    //
    // 这一步不只是为了统一代码路径，更是性能上的关键：f16/bf16 的解码函数
    // （_f16_to_f32 / _bf16_to_f32）定义在另一个编译单元 src/utils/types.cpp 中，
    // 而本项目没有开启 LTO/LTCG，所以它们无法被内联。如果放在矩阵乘的最内层循环里调用，
    // 大测例会产生 86 亿次跨模块函数调用，慢到无法接受。
    // 先一次性解码（只需 m*k + n*k 次调用），之后的乘加就全部在纯 float 上进行。
    //
    // f32 时数据本身就是 float，直接借用原指针，零拷贝、零转换开销。
    std::vector<float> in_buf, w_buf, bias_buf;
    const float *in_f = nullptr;
    const float *w_f = nullptr;
    const float *bias_f = nullptr;

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32: {
        // 直接复用三块原始内存。
        in_f = reinterpret_cast<const float *>(in->data());
        w_f = reinterpret_cast<const float *>(weight->data());
        if (has_bias) {
            bias_f = reinterpret_cast<const float *>(bias->data());
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *ip = reinterpret_cast<const bf16_t *>(in->data());
        const bf16_t *wp = reinterpret_cast<const bf16_t *>(weight->data());
        in_buf.resize(m * k);
        w_buf.resize(n * k);
        for (size_t i = 0; i < m * k; ++i) {
            in_buf[i] = utils::cast<float>(ip[i]); // bf16 -> float
        }
        for (size_t i = 0; i < n * k; ++i) {
            w_buf[i] = utils::cast<float>(wp[i]);
        }
        in_f = in_buf.data();
        w_f = w_buf.data();
        if (has_bias) {
            const bf16_t *bp = reinterpret_cast<const bf16_t *>(bias->data());
            bias_buf.resize(n);
            for (size_t j = 0; j < n; ++j) {
                bias_buf[j] = utils::cast<float>(bp[j]);
            }
            bias_f = bias_buf.data();
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const fp16_t *ip = reinterpret_cast<const fp16_t *>(in->data());
        const fp16_t *wp = reinterpret_cast<const fp16_t *>(weight->data());
        in_buf.resize(m * k);
        w_buf.resize(n * k);
        for (size_t i = 0; i < m * k; ++i) {
            in_buf[i] = utils::cast<float>(ip[i]); // fp16 -> float
        }
        for (size_t i = 0; i < n * k; ++i) {
            w_buf[i] = utils::cast<float>(wp[i]);
        }
        in_f = in_buf.data();
        w_f = w_buf.data();
        if (has_bias) {
            const fp16_t *bp = reinterpret_cast<const fp16_t *>(bias->data());
            bias_buf.resize(n);
            for (size_t j = 0; j < n; ++j) {
                bias_buf[j] = utils::cast<float>(bp[j]);
            }
            bias_f = bias_buf.data();
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }

    // 结果先存放在 float 缓冲区中，最后统一转换回原始数据类型。
    std::vector<float> out_f(m * n);

    // 第 2 步：计算 out[i][j] = sum_p in[i][p] * weight[j][p] + bias[j]。
    //
    // 数学式 Y = X * W^T 中的转置，靠“按行读取 weight”隐式完成：
    // W^T 的第 j 列就是 W 的第 j 行，所以 out[i][j] 等于 in 第 i 行与 weight 第 j 行的点积。
    // 这样两个操作数在内存上都是连续的，缓存友好，也不需要真的搬运数据做转置。
    //
    // 分块大小：一次处理 IB 行输入。作用是提升 weight 的缓存复用率——内层对同一行 weight
    //（k 个 float）连续做 IB 次点积，该行只需从内存读入一次，就被 IB 行输入共用。
    // 于是 weight 的总内存流量从 m*n*k 降到 (m/IB)*n*k，在 512x4096x4096 这组用例上
    // 约由 34GB 降到 4GB，避免整个算子退化成受内存带宽限制。
    constexpr size_t IB = 8;

    for (size_t i0 = 0; i0 < m; i0 += IB) {        // 遍历输入行的分块
        const size_t i_end = std::min(i0 + IB, m); // 本块的结束行（尾块可能不足 IB 行）
        for (size_t j = 0; j < n; ++j) {           // 遍历输出列，即 weight 的行
            const float *w_row = w_f + j * k;      // weight 第 j 行的首地址，长度 k
            for (size_t i = i0; i < i_end; ++i) {  // 本块内的每一行输入
                const float *in_row = in_f + i * k; // in 第 i 行的首地址，长度 k

                // 用 4 个独立累加器做点积。
                // 浮点加法不满足结合律，编译器在没有 /fp:fast 时不能自行拆分归约链；
                // 手动拆成 4 条独立的依赖链，可以让 CPU 流水线并行执行乘加，
                // 明显快于单累加器版本。数值上等价于一种固定的分组求和顺序，
                // 精度不低于（通常略优于）朴素顺序累加。
                float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
                size_t p = 0;
                for (; p + 4 <= k; p += 4) { // 主循环：每次消耗 4 个元素
                    s0 += in_row[p + 0] * w_row[p + 0];
                    s1 += in_row[p + 1] * w_row[p + 1];
                    s2 += in_row[p + 2] * w_row[p + 2];
                    s3 += in_row[p + 3] * w_row[p + 3];
                }
                float sum = (s0 + s1) + (s2 + s3); // 合并 4 条累加链
                for (; p < k; ++p) {               // 处理 k 不是 4 的倍数时剩下的尾部
                    sum += in_row[p] * w_row[p];
                }

                if (bias_f != nullptr) { // 不提供偏置时为空指针，跳过累加
                    sum += bias_f[j];    // 偏置按输出列索引取，广播到所有行
                }
                out_f[i * n + j] = sum;
            }
        }
    }

    // 第 3 步：把 float 结果转换回原始数据类型写出（整个过程只在这里舍入一次）。
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32: {
        float *op = reinterpret_cast<float *>(out->data());
        for (size_t i = 0; i < m * n; ++i) {
            op[i] = out_f[i]; // 目标就是 float，直接写入
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        bf16_t *op = reinterpret_cast<bf16_t *>(out->data());
        for (size_t i = 0; i < m * n; ++i) {
            op[i] = utils::cast<bf16_t>(out_f[i]); // float -> bf16
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        fp16_t *op = reinterpret_cast<fp16_t *>(out->data());
        for (size_t i = 0; i < m * n; ++i) {
            op[i] = utils::cast<fp16_t>(out_f[i]); // float -> fp16
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
