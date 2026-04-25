#!/bin/bash

# ==========================================================
# 模型架构配置 (DeepSeek MoE 7B MVP 版本)
# ==========================================================
NUM_LAYERS=28           # 28 层，方便 PP=2 时平分（每阶段 14 层）
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=11008 

# MoE 专项配置
NUM_EXPERTS=64          # 总路由专家数 (DeepSeek 细粒度设计)
MOE_TOP_K=6             # 激活专家数
NUM_SHARED_EXPERTS=2    # 共享专家数 (始终激活)

# ==========================================================
# 3D 并行策略 (针对单机 4 卡 A100 80GB 深度优化)
# ==========================================================
# 在 4 卡环境下，TP 设为 1 是为了完全消除层内的 All-Reduce 通信，
# 将带宽全部留给 MoE 的 All-to-All 交换。
TP=1                    # Tensor Parallel
PP=2                    # Pipeline Parallel: 分成 2 个 Stage
DP=2                    # Data Parallel: 配合 ZeRO-2 
EP=4                    # Expert Parallel: 4 张卡每张负责 16 个专家

# ==========================================================
# 训练超参数
# ==========================================================
MICRO_BATCH_SIZE=4      # 80GB 显存很充裕，可以适当拉大 MBS 提升吞吐
GLOBAL_BATCH_SIZE=128   # GBS = MBS * DP * Gradient_Accumulation_Steps
SEQ_LEN=2048