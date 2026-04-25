#!/bin/bash

# ==========================================================
# 1. 加载模型架构与并行配置
# ==========================================================
# 确保你已经创建了 configs/model_7b_moe.sh 并写入了 TP=1, PP=2, EP=4 等参数
source configs/model_7b_moe.sh

# ==========================================================
# 2. 分布式环境配置 (单机 4 卡)
# ==========================================================
# 虽然是单机，但保留这些参数可以方便未来扩展到多机
NODE_RANK=0
MASTER_ADDR="localhost" 
MASTER_PORT=6000

# ==========================================================
# 3. 路径配置 (需与服务器实际路径对应)
# ==========================================================
DATA_PATH="data/binary/my_deepseek_pretrain_text_document"
TOKENIZER_PATH="data/tokenizer"
CHECKPOINT_PATH="checkpoints/deepseek_7b_moe"

# ==========================================================
# 4. 启动分布式训练
# ==========================================================
export NCCL_DEBUG=INFO

# 使用 deepspeed 启动
deepspeed pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --moe-expert-parallel-size $EP \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTN_HEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 1000 \
    --lr 2e-4 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 10 \
    --eval-interval 100 \
    --data-path $DATA_PATH \
    --vocab-file $TOKENIZER_PATH/vocab.json \
    --merge-file $TOKENIZER_PATH/merges.txt \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --split 949,50,1 \
    --is-deepseek-moe \
    --num-experts $NUM_EXPERTS \
    --num-shared-experts $NUM_SHARED_EXPERTS \
    --topk $MOE_TOP_K \
    --deepspeed \
    --deepspeed_config ds_config_deepseek_moe.json \
    --zero-stage 2 \
    --fp16 \
    --use-wandb \
    --wandb-project "DeepSeek-7B-MoE-Training" \
    --wandb-exp-name "4xA100-SXM-TP1-PP2-EP4" \
    --wandb-save-interval 1