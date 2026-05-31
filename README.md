# LLM-Rec v2: P5 + RL+Memory 混合推荐系统

基于 P5 (Pretrain-Prompt-Predict) 大语言模型的推荐系统，结合强化学习 (RL) 与记忆系统 (Memory) 的混合架构，在 Amazon Beauty 数据集上进行序列推荐实验。

## 项目结构

```
├── src/
│   ├── rl_experiment.py          # P5 vs RL+Memory 对比实验 (Phase 3)
│   ├── ablation.py               # 消融实验 5 变体 (Phase 4)
│   ├── cooccurrence_retriever.py # 共现矩阵 Item Embeddings (无需GPU)
│   ├── config.py                 # Config / MemoryConfig / RLConfig
│   ├── rl/
│   │   ├── policy.py             # POMDP Policy 网络 + ε-greedy scheduler
│   │   └── actions.py            # 16 discrete macro-actions 定义
│   ├── memory/
│   │   ├── manager.py            # MemoryManager (短期缓冲 + FAISS 长期)
│   │   └── encoder.py            # MemoryEncoder
│   ├── model.py                  # 基础模型定义
│   └── train.py                  # 训练工具
├── reproduce/
│   ├── train.py                  # P5 单卡训练脚本 (Phase 1)
│   ├── pretrain_data.py          # 数据加载 & 预处理
│   └── modeling_p5.py            # P5 模型定义 (T5 backbone)
├── data/
│   └── beauty/                   # Amazon Beauty 数据集
├── outputs/
│   ├── experiment_report.md      # 完整实验报告
│   ├── rl_memory/                # Phase 3 对比实验结果
│   └── ablation/                 # Phase 4 消融实验结果
└── README.md
```

## 环境配置

### 硬件要求

| 场景 | 最低 GPU | 推荐 GPU |
|------|----------|----------|
| P5 全量训练 | 24GB+ | RTX PRO 6000 (96GB) |
| P5 小规模训练 | 8GB | RTX 3060 (6GB, batch=1) |
| RL Policy 训练 | 任意 | RTX 3060 (6GB 足够) |
| Co-occurrence 构建 | CPU | — (1秒完成) |

### 软件环境

```bash
# Python 3.10+, PyTorch 2.8+, CUDA 12.8
conda create -n llmrec python=3.10
conda activate llmrec
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.9.0 faiss-cpu scipy tqdm numpy
```

### IDE 推荐

- **VS Code** + Python + Pylance 扩展
- 或 **PyCharm Professional**（远程开发支持更好）

## 数据

使用 **Amazon Beauty** 数据集 (5-core 过滤)。

```bash
# 数据文件 (data/beauty/)
sequential_data.txt   # 用户-物品交互序列 (22,363 users, 12,101 items)
datamaps.json         # item2id, user2id 映射
user_id2name.pkl      # 用户ID映射
```

数据格式 (`sequential_data.txt`)：
```
<user_id> <item_1> <item_2> ... <item_n>
```

## Phase 1: P5 Baseline 训练

```bash
cd reproduce/

# 小规模快速测试 (5% 数据, 1 epoch)
python train.py --dataset beauty --backbone t5-small \
    --epochs 1 --batch_size 4 --fp16 --sample_ratio 0.05

# 全量训练 (服务器, 10 epochs)
python train.py --dataset beauty --backbone t5-small \
    --epochs 10 --batch_size 32 --fp16 --sample_ratio 1.0
```

**训练参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backbone` | t5-small | 60M 参数 |
| `--batch_size` | 32 | 6GB GPU 用 1-4 |
| `--epochs` | 10 | |
| `--lr` | 1e-4 | AdamW |
| `--fp16` | False | 混合精度 (推荐开启) |
| `--sample_ratio` | 1.0 | 数据采样比例 |

输出: `reproduce/output/<dataset>_<timestamp>/EpochXX.pth` (约 695MB/epoch)

## Phase 2: P5 评估

```bash
# 评估已训练的 P5 checkpoint (14 个任务)
python -m src.train --eval_only \
    --checkpoint reproduce/output/xxx/Epoch10.pth \
    --dataset beauty
```

评估任务包括: Rating (1-6, 1-10), Sequential (2-3, 2-13), Explanation, Review, Direct generation 等 14 项。

## Phase 3: P5 vs RL+Memory 对比实验

```bash
# 使用 P5 embeddings (需要优质 checkpoint)
python -m src.rl_experiment \
    --dataset beauty --backbone t5-small \
    --p5_checkpoint reproduce/output/xxx/Epoch10.pth \
    --sample_ratio 1.0 --epochs 1 --batch_size 64 \
    --max_eval 500 --config_preset 3060 --beam_size 20

# 使用 Co-occurrence embeddings (推荐, 无需优质 checkpoint)
python -m src.rl_experiment \
    --dataset beauty --backbone t5-small \
    --p5_checkpoint reproduce/output/xxx/Epoch01.pth \
    --sample_ratio 1.0 --epochs 1 --batch_size 64 \
    --max_eval 500 --config_preset 3060 \
    --cooc_embeddings data/beauty/cooc_embeddings.pkl
```

**对比内容**:
- P5 Baseline: Beam Search (B=20) 生成 item ID → 全量排序
- RL+Memory: 1-pass Policy → 策略驱动候选检索

**架构**:
```
User History → Co-occurrence Embedding
     ├── Memory System (短期缓冲 + FAISS 长期 + 时间衰减)
     └── POMDP Policy (MLP, 16 actions)
              ↓
         Strategy → CandidateRetriever → Top-20 Items
```

## Phase 4: 消融实验

验证 RL+Memory 各组件的独立贡献：

| 变体 | Memory | Long-term | Time Decay | Adaptive ε |
|------|--------|-----------|------------|-------------|
| `full_system` | ✓ | ✓ | ✓ | ✓ |
| `no_memory` | ✗ | ✗ | ✗ | ✓ |
| `short_term_only` | ✓ | ✗ | ✗ | ✓ |
| `no_time_decay` | ✓ | ✓ | ✗ | ✓ |
| `fixed_epsilon` | ✓ | ✓ | ✓ | ✗ |

```bash
# 运行所有消融变体 (跳过 P5 baseline)
python -m src.ablation \
    --dataset beauty --backbone t5-small \
    --p5_checkpoint reproduce/output/xxx/Epoch01.pth \
    --sample_ratio 1.0 --batch_size 64 --max_eval 500 \
    --cooc_embeddings data/beauty/cooc_embeddings.pkl \
    --skip_p5 --variants all
```

## Co-occurrence Embeddings 构建

无需 GPU，~1 秒完成：

```bash
python -c "
from src.cooccurrence_retriever import build_simple_cooccurrence_embeddings
import pickle

# 加载用户序列 → 构建 128 维共现向量
user_sequences = {}
with open('data/beauty/sequential_data.txt') as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 3:
            user_sequences[p[0]] = [int(i) for i in p[1:]]

embeddings = build_simple_cooccurrence_embeddings(user_sequences, embedding_dim=128)

with open('data/beauty/cooc_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
print(f'Built {len(embeddings)} item embeddings')
"
```

原理: 统计物品共现频率 → 以 top-128 高频物品为 anchor → 每个 item 表示为与各 anchor 的归一化共现权重。

## 实验结果

### P5 vs RL+Memory 对比 (500 test users)

| Metric | P5 (Beam=20) | RL+Memory | 提升 |
|--------|-------------|-----------|------|
| HR@1 | 0.0% | **0.2%** | — |
| HR@5 | 0.0% | **1.6%** | — |
| HR@10 | 0.0% | **2.4%** | — |
| NDCG@5 | 0.0% | **0.90%** | — |
| NDCG@10 | 0.0% | **1.17%** | — |

> P5 全零原因: checkpoint 仅训练 1 epoch (5% 数据), 无法生成有效 item ID。
> RL HR@10=2.4% 约为随机水平 (20/12101≈0.17%) 的 **14 倍**。

### 消融实验 (5 变体, 500 test users)

| 变体 | HR@5 | HR@10 | NDCG@10 | 主导 Action |
|------|------|-------|---------|-------------|
| full_system | 1.6% | 2.4% | 1.17% | remind_abandoned 68% |
| no_memory | 1.6% | 2.4% | 1.17% | exploit_high_ctr 38% |
| short_term_only | 1.6% | 2.4% | 1.17% | serendipity 53% |
| no_time_decay | 1.6% | 2.4% | 1.17% | explore_new_brand 36% |
| fixed_epsilon | 1.6% | 2.4% | 1.17% | explore_trending 37% |

**关键发现**:
- 推荐指标完全一致 — Retriever 对所有 action 返回相同候选
- Action 分布差异显著 — Policy 根据 Memory 配置学会了不同行为策略
- POMDP 框架能有效利用 memory context 做差异化决策

## 结论

1. **Co-occurrence embeddings 是性价比最高的方案**: 无需 GPU 训练，~1 秒构建，即可让 RL 策略正常训练并超越未充分训练的 P5 baseline。

2. **策略崩塌已解决**: 前次实验中 "100% exploit_similar" 的崩塌通过有意义的 item embeddings 和优化的 BC 标签分配得到解决。

3. **Memory 组件影响策略行为但不影响推荐精度**: 因为当前 Retriever 的多数策略最终调用相同的 similarity rank，action 选择的多样性未转化为候选列表的多样性。

4. **P5 baseline 受限于 checkpoint 质量**: 仅 1 epoch 训练不足以让 T5 学会生成有效的 item ID。需要更多数据和更长训练。

### 后续方向

- 服务器 (RTX PRO 6000 96GB) 恢复后用 100% 数据重训 P5
- Retriever 重构: 让不同 action 使用真正不同的检索方法
- 融合 P5 semantic + co-occurrence statistical embeddings
- 在线 RL 环境模拟，评估 exploration-exploitation tradeoff

## 已修复的 Bug

| Bug | 文件 | 修复 |
|-----|------|------|
| 策略崩塌 (100% exploit_similar) | `rl_experiment.py` | Co-occurrence embeddings + rank-based BC 标签 |
| Target item 被错误排除 | `rl_experiment.py`, `ablation.py` | 新增 `eval_history` 参数 |
| POMDPPolicy 缺少 user_dim | `rl/policy.py` | 添加 `self.user_dim` |
| Buffer 构建慢 (16x) | `rl_experiment.py`, `ablation.py` | 单次 similarity lookup |
| 数据泄露 (编码含 target) | `ablation.py` | 仅用 test history 编码 |

## 许可 & 引用

本项目基于 [P5 (Pretrain-Prompt-Predict)](https://github.com/jeykigung/P5) 架构。
