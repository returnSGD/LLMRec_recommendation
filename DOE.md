# DOE: 实验设计方案 —— RL + 长时序记忆推荐系统 vs P5 Baseline

## 0. 基线模型概述 (P5)

P5 (Recommendation as Language Processing) 将推荐系统统一为文本生成任务，基于 T5 encoder-decoder 架构：

- **5 个任务族**: rating (9 prompt)、sequential (12 prompt)、explanation (11 prompt)、review (3 prompt)、traditional / direct recommendation (7 prompt)
- **训练方式**: 多任务联合预训练，损失函数为语言建模负对数似然，`losses='rating,sequential,review,metadata,recommend'`
- **推理方式**: Beam search (B=20) 做序列推荐候选生成，全物品空间评估
- **数据限制**: 输入截断 512 token，无长期记忆，无负反馈区分
- **数据集**: Amazon (Beauty, Toys, Sports) + Yelp
- **数据划分**: 序列推荐 leave-last-out，其他任务 80/10/10
- **评估指标**: RMSE/MAE (评分), HR@k/NDCG@k (序列 & 直接推荐), BLEU-4/ROUGE-1/2/L (解释 & 评论)
- **参数量**: P5-S: 60.75M; P5-B: 223.28M

---

## 1. 实验一：复现 P5 Baseline

### 1.1 目标

在 Amazon Beauty 数据集上完整复现 P5 (Pretrain, Personalized Prompt & Predict Paradigm) 全部 5 个任务族 + 零样本跨域迁移评估，建立可靠的性能基准，验证复现结果与 P5 论文 (RecSys 2022) 报告值一致。

### 1.2 P5 架构回顾 (基于 baseline_model 源码分析)

#### 1.2.1 模型组成

```
P5 = T5ForConditionalGeneration + Whole-Word Embedding (JointEncoder)
```

- **JointEncoder** (`modeling_p5.py:L23`): 继承 T5Stack，增加 `whole_word_embeddings` (nn.Embedding(512, d_model))，将连续子词 token 映射到同一 whole-word ID，encoder 输入 = token_embed + whole_word_embed + position_bias
- **Decoder**: 标准 T5Stack，无额外修改
- **LM Head**: 线性映射 d_model → vocab_size，与 shared embedding 权重绑定 (tie_word_embeddings=True)
- **Tokenization**: P5Tokenizer (继承 T5Tokenizer)，SentencePiece vocab_size=32,128，user_id/item_id 以 "user_XXX"/"item_XXX" 形式分词为多个 sub-word 单元 (而非独立 extra token)

关键参数量:
| 组件 | P5-Small | P5-Base |
|------|----------|---------|
| Encoder 层数 | 6 | 12 |
| Decoder 层数 | 6 | 12 |
| d_model | 512 | 768 |
| Attention heads | 8 | 12 |
| Feedforward dim | 2048 | 3072 |
| 总参数量 | 60.75M | 223.28M |

#### 1.2.2 训练目标

统一语言建模负对数似然 (NLL)，所有任务共享同一损失函数:

$$
\mathcal{L}_{\theta}^{P5} = -\sum_{j=1}^{|y|} \log P_{\theta}(y_j \mid y_{<j}, \mathbf{x})
$$

在 `pretrain_model.py:P5Pretraining.train_step()` 中实现: 对每个 batch 按 task 分别累计 loss，支持 `losses='rating,sequential,explanation,review,traditional'` 多任务 loss 分别记录。

#### 1.2.3 推理方式

| 任务族 | 解码方式 | Beam Size | 备注 |
|--------|---------|-----------|------|
| Rating prediction | Greedy | 1 | 直接生成评分数字 |
| Sequential recommendation | Beam search | 20 | 全物品空间 (all-item setting) |
| Explanation | Greedy | 1 | 文本生成 |
| Review summarization | Greedy | 1 | 文本生成 |
| Direct recommendation | Beam search / Softmax prob | 20 | 候选集 100 物品 (1 positive + 99 negative) |

对于 Direct Recommendation 中的 yes/no 判别式 prompt (5-1 ~ 5-4)，使用 "yes" token 的 softmax 生成概率对所有候选物品排序；对于开放式生成 prompt (5-5 ~ 5-8)，使用 beam search (B=20) 生成候选列表。

### 1.3 数据预处理管线

#### 1.3.1 原始数据 → P5 格式

数据预处理在 `baseline_model/preprocess/` 中，需要执行以下步骤:

**Step 1: 数据分割**
```
data/beauty/
├── review_splits.pkl      # 评分/评论/解释 80/10/10 划分
├── exp_splits.pkl         # 解释数据 (从 review 中提取 feature word + 解释句)
├── rating_splits_augmented.pkl  # 高斯采样增强后的评分数据
├── sequential_data.txt    # 用户交互序列 (每行: user_id item1 item2 ...)
├── negative_samples.txt   # 测试用负样本
├── datamaps.json          # user2id, item2id, id2item 映射
├── user_id2name.pkl       # 用户名映射 (用于个性化 prompt)
├── meta.json.gz           # 物品元数据 (title, price, brand, etc.)
└── zeroshot_exp_splits.pkl # 零样本跨域迁移评估数据
```

**Step 2: P5 数据加载** (`pretrain_data.py`)

`P5_Amazon_Dataset` 类负责将预处理数据转换为 input-target token 序列:
- `compute_datum_info()`: 根据 `train_sample_numbers` 规划每个 task group 使用哪些数据样本
- `__getitem__()`: 随机选择一个 prompt template，用数据字段填充 → source_text + target_text → tokenize → (input_ids, whole_word_ids, target_ids)
- `collate_fn()`: 动态 padding 到 batch 内最大长度
- `gaussian_sampling()`: 训练时将离散评分 (1-5) 通过高斯采样增强为连续值 (41 个类别)

**Step 3: 数据采样策略** (`pretrain.py:L360-361`)

```python
# Amazon 数据集
train_task_list = {
    'rating': ['1-1'~'1-9'],
    'sequential': ['2-1'~'2-12'],
    'explanation': ['3-1'~'3-11'],
    'review': ['4-1', '4-2', '4-3'],
    'traditional': ['5-1'~'5-7']
}
train_sample_numbers = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1, 'traditional': (10, 5)}
```

sequential 的 (5,5,10) 含义: 直接预测下一物品 (2-1~2-6, 2-13) 每样本重复 5 次, 从候选列表选择 (2-7~2-10) 重复 5 次, yes/no 判断 (2-11~2-12) 重复 10 次。此设计使得不同 prompt group 在训练中均衡出现。

### 1.4 42 个 Personalized Prompt 模板

来自 `baseline_model/src/all_amazon_templates.py`:

#### Rating Prediction (10 prompts: 1-1 ~ 1-10)

| ID | Input Template | Target | 类型 |
|----|---------------|--------|------|
| 1-1 | "Given the user {user_id} and item {item_id}, what rating will user give?" | "{rating}" | 直接预测 |
| 1-2 | "What star rating will user_{user_id} give for this item: {item_title}?" | "{rating}" | 物品标题 |
| 1-3 | "Does user_{user_id} think item_{item_id} deserves a rating of {score}?" | "yes"/"no" | 判别式 |
| 1-4 | "How does user_{user_id} rate this product: item_{item_id}?" | "like"/"dislike" | 二分类 |
| 1-5 | "Given the user {user_id} and item {item_id} {item_title}, give a rating." | "{rating}" | 物品标题 |
| 1-6 | "User {user_name} and item {item_id}. Predict the rating score." | "{rating}" | 用户名 |
| 1-7 | "What is the rating score that {user_name} will give to product {item_title}?" | "{rating}" | 用户名+标题 |
| 1-8 | "Does {user_name} think the product {item_title} merits a score of {score}?" | "yes"/"no" | 用户名+标题+判别 |
| 1-9 | "Does {user_name} like or dislike the product {item_title}?" | "like"/"dislike" | 用户名+标题+二分类 |
| 1-10 | "Which star rating will {user_name} give item: {item_title}?" | "{rating}" | ★零样本评估 |

#### Sequential Recommendation (13 prompts: 2-1 ~ 2-13)

| ID | Input | Target | 类型 |
|----|-------|--------|------|
| 2-1 | "Given the purchase history of user_{uid}: {history}, what is the next item?" | {item_id} | 直接预测 |
| 2-2 | "Based on user_{uid}'s interaction history: {history}, predict the next likely purchase." | {item_id} | 直接预测 |
| 2-3 | "Here is user_{uid}'s purchase history: {history}. Try to recommend the next item." | {item_id} | ★评估用 |
| 2-4 | "{user_name} purchased the following items: {history}. Predict the next item." | {item_id} | 用户名 |
| 2-5 | "The purchasing record of {user_name}: {history}, give the next recommendation." | {item_id} | 用户名 |
| 2-6 | "What is the next likely item that {user_name} will buy after seeing {history}?" | {item_id} | 用户名 |
| 2-7 | "From the candidate list: {candidates}, what will user_{uid} buy next given {history}?" | {item_id} | 候选列表 |
| 2-8 | "{user_name}, pick one from {candidates} as the next after {history}." | {item_id} | 用户名+候选 |
| 2-9 | "Given {history}, what item from {candidates} will user_{uid} likely purchase?" | {item_id} | 候选列表 |
| 2-10 | "From the following candidates {candidates}, what will {user_name} buy?" | {item_id} | 用户名+候选 |
| 2-11 | "Does user_{uid} tend to buy {item} after {history}?" | "yes"/"no" | 判别式 |
| 2-12 | "Will {user_name} buy {item} given purchase record {history}?" | "yes"/"no" | 用户名+判别 |
| 2-13 | "I find {user_name}'s purchase history list: {history}. I wonder what to buy next." | {item_id} | ★零样本评估 |

#### Explanation Generation (12 prompts: 3-1 ~ 3-12)

| ID | Input | Target | 类型 |
|----|-------|--------|------|
| 3-1 | "Generate an explanation for user_{uid} about this product: {item_title}." | {explanation} | 物品标题 |
| 3-2 | "Given headline: {summary}, help user_{uid} describe item_{iid}." | {explanation} | 评论摘要 |
| 3-3 | "Help user_{uid} explain why they rate item {item_title} with {rating} stars." | {explanation} | ★评估用 |
| 3-4 | "Generate a text explanation for {user_name} and item {item_title}." | {explanation} | 用户名+标题 |
| 3-5 | "The post headline: {summary}. {user_name} writes an explanation for {item_title}." | {explanation} | 用户名+摘要 |
| 3-6 | "{user_name} gave item_{iid} {rating} stars. Write an explanation sentence." | {explanation} | 用户名+评分 |
| 3-7 | "Based on the word **{feature}**, help user_{uid} write a {rating}-star explanation for item_{iid}." | "{rating}, {explanation}" | 特征词 hint |
| 3-8 | "Using the feature word {feature}, {user_name} explains item_{iid}." | "{rating}, {explanation}" | 特征词 hint |
| 3-9 | "Feature word: {feature}. User_{uid} comments on {item_title}:" | {explanation} | ★评估用 (特征词) |
| 3-10 | "The given feature word is {feature}. {user_name} writes about {item_title}." | {explanation} | 用户名+特征词 |
| 3-11 | "Feature word {feature}, rating {rating}. User_{uid} talks about item_{iid}:" | {explanation} | 特征词+评分 |
| 3-12 | "The word **{feature}** is important. {user_name} rates {rating} for item_{iid}." | {explanation} | ★零样本评估 |

#### Review Related (4 prompts: 4-1 ~ 4-4)

| ID | Input | Target | 类型 |
|----|-------|--------|------|
| 4-1 | "User_{uid} wrote: {review_text}. Summarize to a short sentence." | "{summary}" | 评论摘要 |
| 4-2 | "Given {user_uid}'s review: {review_text}, predict the rating." | "{rating}" | 评分预测 |
| 4-3 | "{user_name} says: {review_text}. What is the gist of the review?" | "{summary}" | 用户名 |
| 4-4 | "From {user_name}'s comment: {review_text}, what rating will they give?" | "{rating}" | ★零样本评估 |

#### Direct Recommendation (8 prompts: 5-1 ~ 5-8)

| ID | Input | Target | 类型 |
|----|-------|--------|------|
| 5-1 | "Will user_{uid} likely interact with item_{iid}?" | "yes"/"no" | 判别式 |
| 5-2 | "{user_name}, do you want to buy item_{iid}?" | "yes"/"no" | 用户名 |
| 5-3 | "Does {user_name} prefer {item_title}?" | "yes"/"no" | 用户名+标题 |
| 5-4 | "Would user_{uid} recommend {item_title} to other people?" | "yes"/"no" | 判别式 |
| 5-5 | "Select a suitable item from {candidates} to recommend to {user_name}." | {item_id} | 候选选择 |
| 5-6 | "From the candidate list {candidates}, what is the best item for {user_name}?" | {item_id} | 候选选择 |
| 5-7 | "What item from {candidates} do you think user_{uid} is likely to interact with?" | {item_id} | 候选选择 |
| 5-8 | "Among {candidates}, recommend an item for user_{uid} and tell the item ID only." | {item_id} | ★零样本评估 |

#### Zero-shot Cross-domain Transfer (7 prompts: Z-1 ~ Z-7)

| ID | Input | Target | 类型 |
|----|-------|--------|------|
| Z-1 | "User_{uid} encounters a new item: {title}, price {price}, brand {brand}. Like or dislike?" | "like"/"dislike" | 新物品偏好 |
| Z-2 | "Considering {title}, brand {brand}, price {price}, what rating will {user_name} give?" | "{rating}" | 用户评分预测 |
| Z-3 | "How high will user_{uid} rate the product: {title}, {price} USD, from {brand}?" | "{rating}" | 物品评分预测 |
| Z-4 | "Does {user_name} like this new {brand} product: {title}, priced at {price}?" | "like"/"dislike" | 用户偏好二分类 |
| Z-5 | "{user_name} writes an explanation for this product: {title}, {brand}, {price}." | "{explanation}" | 直接解释 |
| Z-6 | "Based on the word **{feature}**, help user_{uid} write a {rating}-star explanation for: {title}, {price}, {brand}." | "{explanation}" | 特征词解释 |
| Z-7 | "What would {user_name} say about this {brand} product: {title}?" | "{explanation}" | 直接解释 |

### 1.5 训练超参数 (与论文严格对齐)

| 配置项 | P5-Small | P5-Base | 说明 |
|--------|----------|---------|------|
| Backbone | t5-small | t5-base | HuggingFace checkpoint |
| Tokenizer | P5Tokenizer (SentencePiece) | 同 | vocab_size=32,128 |
| Epochs | 10 | 10 | 论文 Table 1 脚注 |
| Batch size | 32 | 16 | 对应脚本 `pretrain_P5_small_beauty.sh` |
| Learning rate | 1e-3 (peak) | 1e-3 (peak) | AdamW |
| Warmup | 5% of total iterations | 同 | warmup_ratio=0.05 |
| Weight decay | 0.01 | 0.01 | 仅应用于非 bias/LayerNorm 参数 |
| Adam eps | 1e-6 | 1e-6 | |
| Adam betas | (0.9, 0.999) | 同 | |
| Gradient clip | 1.0 | 1.0 | |
| Dropout | 0.1 | 0.1 | 所有层统一 |
| Max source length | 512 | 512 | token 数 |
| Max target length | 64 | 64 | token 数 |
| Mixed precision | FP16 (Native AMP) | FP16 | |
| GPUs | 4 × NVIDIA RTX A5000 | 同 | 分布式训练 (DistributedDataParallel) |
| Whole word embedding | True | True | 核心架构组件 |
| Losses | rating,sequential,explanation,review,traditional | 同 | 5 任务联合训练 |
| Random seed | 2022 | 2022 | |

### 1.6 执行步骤

#### Phase 1: 环境准备 (预计 1 天)

```bash
# 1. 安装依赖
pip install torch transformers sentencepiece tensorboard

# 2. 验证数据完整性 — 检查 4 个数据集目录
# data/beauty/, data/sports/, data/toys/, data/yelp/
# 每个目录应包含 9 个文件 (见 1.3.1)

# 3. 验证 baseline_model 代码可运行
cd baseline_model
python -c "from src.modeling_p5 import P5; print('P5 module OK')"
python -c "from src.tokenization import P5Tokenizer; print('Tokenizer OK')"
```

#### Phase 2: Amazon Beauty 单数据集复现 (预计 3-5 天训练 + 1 天评估)

```bash
# 训练命令 (基于 baseline_model/scripts/pretrain_P5_small_beauty.sh)
cd baseline_model
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port 12345 \
    src/pretrain.py \
        --distributed --multiGPU \
        --seed 2022 \
        --train beauty \
        --valid beauty \
        --batch_size 32 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-3 \
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses 'rating,sequential,explanation,review,traditional' \
        --backbone 't5-small' \
        --output snap/beauty-small \
        --epoch 10 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --whole_word_embed
```

训练日志将通过 `tqdm` 输出每个 task 的 loss。模型每 epoch 保存一次到 `snap/beauty-small/EpochXX.pth`，最佳验证 loss 保存为 `BEST_EVAL_LOSS.pth`。

**训练监控要点**:
- `total_loss`: 每 epoch 应持续下降, 10 epochs 后趋于收敛
- 各 task loss 的相对比例: sequential 和 traditional 通常较高 (需要 beam search 的任务更复杂)
- 如果 `rating_loss` 在 epoch 5 后不再下降, 说明该简单任务已饱和

#### Phase 3: 全任务评估 (预计 1 天)

评估脚本需要编写 `eval.py`, 对每个任务族执行以下评估:

**A. Rating Prediction 评估**
```
输入 prompt: 1-6 (seen) 和 1-10 (unseen/零样本)
解码: Greedy decoding, 生成评分文本 → parse 为浮点数
指标: RMSE = sqrt(mean((pred - true)^2)), MAE = mean(|pred - true|)
对比基线: MF, MLP (论文 Table 2)
```

**B. Sequential Recommendation 评估**
```
输入 prompt: 2-3 (seen) 和 2-13 (unseen/零样本)
解码: Beam search B=20, 全物品空间 (不采样)
处理: beam 生成的文本 → 提取 item_id → 在全物品中匹配 → 排序
指标: HR@1, HR@5, HR@10, NDCG@5, NDCG@10
对比基线: Caser, HGN, GRU4Rec, BERT4Rec, FDSA, SASRec, S3-Rec (论文 Table 3)
关键: 必须与论文一致使用 "全物品排序" (all-item setting) 而非采样评估
```

**C. Explanation Generation 评估**
```
输入 prompt: 3-3 (直接), 3-9 (特征词 seen), 3-12 (特征词 unseen)
解码: Greedy decoding
指标: BLEU-4 (nltk / sacrebleu), ROUGE-1/2/L (rouge-score)
对比基线: Attn2Seq, NRT, PETER, PETER+ (论文 Table 4)
```

**D. Review Related 评估**
```
输入 prompt: 4-1 (评论摘要 seen), 4-2 (评分预测 seen), 4-4 (评分预测 unseen)
解码: Greedy decoding
指标: BLEU-4, ROUGE-1/2/L (摘要); RMSE, MAE (评分预测)
对比基线: T0, GPT-2 (论文 Table 5 & Table 6)
```

**E. Direct Recommendation 评估**
```
输入 prompt: 5-1, 5-4 (判别式 seen/unseen), 5-5, 5-8 (生成式 seen/unseen)
判别式方法: 对每个候选物品计算 P("yes") → 排序
生成式方法: Beam search B=20 生成候选 → 匹配排序
候选池: 1 positive + 99 negative samples (论文 Table 7)
指标: HR@1, HR@5, HR@10, NDCG@5, NDCG@10
对比基线: BPR-MF, BPR-MLP, SimpleX (论文 Table 7)
```

**F. Zero-shot Cross-domain Transfer 评估**
```
模型: 在源域 (如 Beauty) 预训练的 P5-Base
评估: 使用 Z-1 ~ Z-7 prompt, 在目标域 (Toys/Sports) 的共享用户上评估
指标: 偏好准确率 (Z-1/Z-4), MAE (Z-2/Z-3), BLEU-2/ROUGE-1 (Z-5~Z-7)
对比: 论文 Table 9 的 6 个迁移方向
```

### 1.7 论文精确基准值 (复现目标)

以下为 P5 论文报告的确切数值，复现时偏差 ≤ 2% 视为通过:

#### P5-Small on Amazon Beauty (核心复现目标)

| 任务 | 指标 | P5-Small 论文值 | 对应 Table |
|------|------|----------------|-----------|
| Rating (1-6) | RMSE | 1.3128 | Table 2 |
| Rating (1-6) | MAE | 0.8428 | Table 2 |
| Rating (1-10) | RMSE | 1.2989 | Table 2 |
| Rating (1-10) | MAE | 0.8473 | Table 2 |
| Sequential (2-3) | HR@5 | 0.0503 | Table 3 |
| Sequential (2-3) | NDCG@5 | 0.0370 | Table 3 |
| Sequential (2-3) | HR@10 | 0.0659 | Table 3 |
| Sequential (2-3) | NDCG@10 | 0.0421 | Table 3 |
| Sequential (2-13) | HR@5 | 0.0490 | Table 3 |
| Sequential (2-13) | NDCG@5 | 0.0358 | Table 3 |
| Sequential (2-13) | HR@10 | 0.0646 | Table 3 |
| Sequential (2-13) | NDCG@10 | 0.0409 | Table 3 |
| Explanation (3-3) | BLEU-4 | 1.2237 | Table 4 |
| Explanation (3-3) | ROUGE-1 | 17.6938 | Table 4 |
| Explanation (3-3) | ROUGE-L | 12.8606 | Table 4 |
| Explanation (3-9) | BLEU-4 | 1.9788 | Table 4 |
| Explanation (3-9) | ROUGE-1 | 25.6253 | Table 4 |
| Explanation (3-12) | BLEU-4 | 1.9425 | Table 4 |
| Explanation (3-12) | ROUGE-1 | 25.1474 | Table 4 |
| Review (4-2) | RMSE | 0.6233 | Table 5 |
| Review (4-2) | MAE | 0.3051 | Table 5 |
| Review (4-1) | BLEU-2 | 2.1225 | Table 6 |
| Review (4-1) | ROUGE-1 | 8.4205 | Table 6 |
| Direct (5-4) | HR@1 | 0.0862 | Table 7 |
| Direct (5-4) | HR@5 | 0.2448 | Table 7 |
| Direct (5-4) | NDCG@5 | 0.1673 | Table 7 |
| Direct (5-4) | HR@10 | 0.3441 | Table 7 |
| Direct (5-4) | NDCG@10 | 0.1993 | Table 7 |
| Direct (5-5) | HR@1 | 0.0601 | Table 7 |
| Direct (5-5) | HR@5 | 0.1611 | Table 7 |
| Direct (5-5) | NDCG@5 | 0.1117 | Table 7 |
| Direct (5-5) | HR@10 | 0.2370 | Table 7 |
| Direct (5-8) | HR@1 | 0.0571 | Table 7 |
| Direct (5-8) | HR@5 | 0.1566 | Table 7 |
| Direct (5-8) | NDCG@5 | 0.1078 | Table 7 |
| Direct (5-8) | HR@10 | 0.2317 | Table 7 |

#### P5-Base on Amazon Beauty (可选，更大模型复现)

| 任务 | 指标 | P5-Base 论文值 | 对应 Table |
|------|------|---------------|-----------|
| Sequential (2-3) | HR@5 | 0.0508 | Table 3 |
| Sequential (2-3) | HR@10 | 0.0664 | Table 3 |
| Direct (5-1) | HR@10 | 0.1593 | Table 7 |
| Direct (5-8) | HR@10 | 0.2300 | Table 7 |

### 1.8 验收标准

1. **主要标准**: P5-Small on Beauty 的 30 个指标中, 与论文值偏差 ≤ 2% 的指标 ≥ 25 个
2. **次要标准**: 所有指标的趋势方向与论文一致 (如 Sequential HR@10 > Direct HR@10)
3. **复现通过后**: 使用相同代码在 Toys, Sports, Yelp 上批量训练, 确认跨数据集一致性
4. **失败处理**: 如果某个任务族指标系统性偏低 >5%, 检查:
   - 该任务的数据划分是否与论文一致
   - Beam search 的 B 值是否正确 (必须是全物品空间, 不能采样)
   - 该任务的 prompt template 是否正确 (检查 `all_amazon_templates.py`)
   - Tokenization 是否正确处理了 user/item ID 的 whole-word 映射

### 1.9 关键风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| 分布式训练 DDP 配置错误 | 中 | 训练失败或极慢 | 先用单 GPU 小 batch 验证 pipeline 正确性 (`--batch_size 4`) |
| 数据路径硬编码不匹配 | 中 | FileNotFoundError | 检查 `pretrain_data.py` 中所有 `os.path.join('data', ...)` 路径 |
| FAISS 依赖未安装 | 低 | — | P5 基线复现不需要 FAISS，仅本系统需要，互不干扰 |
| 全物品排序内存溢出 | 中 | 评估阶段 OOM | 使用分块计算 + CPU offload，或先在小数据集 (Toys: 11,924 items) 上验证 |
| 复现结果与论文值偏差 >5% | 中 | 基线不可靠 | 逐一检查: (1) 负样本编码是否与论文一致 (2) Gaussian sampling 是否正确 (3) 训练 epoch 是否足够 (4) Beam size=20 是否正确设置 |
| FP16 混合精度数值不稳定 | 低 | loss NaN | 检查 gradient clip=1.0 是否正确应用, 降低 lr 到 5e-4 |

## 2. 实验二：主对比实验

### 2.1 目标

在严格控制所有公平性变量的条件下，对比我们的完整系统 (RL + 层次化记忆 + 不确定性自适应探索) 与 P5 Baseline 的全面性能。

### 2.2 公平性控制

#### 2.2.1 必须相同 (Fair Comparison 底线)

| 控制项 | 具体内容 | 原因 |
|--------|----------|------|
| **数据集** | Amazon (Sports, Beauty, Toys) + Yelp，与 P5 完全相同 | 不同数据集的结果不可比较 |
| **训练/验证/测试划分** | 评分/解释/评论任务: 80/10/10 随机划分；序列推荐: 留最后一件做测试、倒数第二件做验证；直接推荐: 遵循序列推荐的训练划分 | P5 的划分策略影响性能，必须一致 |
| **评估指标** | 评分: RMSE, MAE；序列 & 直接推荐: HR@k, NDCG@k；解释 & 评论: BLEU-4, ROUGE-1/2/L；零样本: 同上 | 不同指标不可直接对比 |
| **物品池** | 全物品设置 (非采样评估)，与 P5 一致 | 采样评估会虚高指标 |
| **评估协议** | 序列推荐: Beam search (B=20) 生成候选列表，在全物品空间评估；直接推荐: 从候选集中选最优 | P5 的核心推理协议 |
| **任务定义** | 评分 (1-5 分预测 & 喜欢/不喜欢)、序列推荐 (下一物品预测)、解释生成、评论摘要、直接推荐 (top-k)，覆盖 P5 全部 5 个任务族 | 不同任务定义无法比较 |
| **T5 主干初始化** | 使用相同 T5-small / T5-base HuggingFace 预训练检查点 | 保证语言理解能力的起点一致 |

#### 2.2.2 应该报告但可以有差异 (Trade-off 讨论)

| 控制项 | P5 取值 | 我们的处理 | 说明 |
|--------|--------|------------|------|
| 模型总参数量 | P5-S: 60.75M; P5-B: 223.28M | 诚实报告记忆编码器 + RL 策略网络的额外参数，设参数量消融组 | 需讨论"性能提升是否单纯来自更多参数" |
| 最大输入长度 | 512 tokens | 记忆系统的长期信息不受 512 token 限制 (这是我们的优势)，同时报告 P5 在 512/1024/2048 窗口下的性能作为 sensitivity analysis | 如果 P5 在更大窗口下也能追上，则记忆系统优势减弱 |
| 训练 epoch | 10 epochs | 应报告训练计算量 (GPU-hours) | 效率 trade-off |
| 预训练数据 | T5 原始预训练权重 | 相同 T5 checkpoint | 确保语言知识基线相同 |

#### 2.2.3 可以有差异 (核心卖点)

| 差异项 | P5 | 本系统 | 说明 |
|--------|-----|--------|------|
| 可用历史长度 | 512 token 窗口截断 | 外部记忆库存储全量历史 | 这正是我们要验证的优势 |
| 训练目标 | NLL | RL 累积奖励最大化 | 不同优化范式 |
| 推理方式 | Beam search (B=20) | 离散动作选择 (1 次前向) | 推理效率贡献 |
| 负反馈建模 | 无 | 显式建模 | 单独做消融 |

### 2.3 对比组设计

| 组 | 模型 | 说明 |
|----|------|------|
| **A** | P5-small (原始) | 512 token 窗口, beam=20, 官方配置复现 |
| **B** | P5-small + 长窗口 | 1024 / 2048 token 窗口 sensitivity analysis |
| **C** | **完整系统** | RL + 层次化记忆 + 不确定性自适应探索 + LLM 渲染 |
| **D** | 完整系统 - 渲染层 | 去掉 LLM 解释生成，纯 RL 决策 (用于效率测试) |
| **E** | P5-base (原始) | 223.28M 参数基线，用于参数量 fairness 讨论 |

### 2.4 预期结果矩阵

| 任务类型 | 指标 | P5 (A) | 本系统 (C) | 预期差异 | 对应的核心贡献 |
|----------|------|--------|-----------|----------|---------------|
| Sequential (普通) | HR@10 | baseline | comparable (±2%) | — |
| Sequential (长周期用户) | HR@10 | 低 | **显著高** (↑10-20%) | 核心贡献一 |
| Direct Recommendation | NDCG@10 | baseline | comparable (±2%) | — |
| 多样性 (30天仿真) | Diversity@20 | 低 | **显著高** (↑15-25%) | 核心贡献二 |
| Rating | RMSE | baseline | comparable (±2%) | — |
| Explanation | BLEU-4 | baseline | comparable (±2%) | 保持生成能力 |
| Review | BLEU-4 | baseline | comparable (±2%) | 保持生成能力 |
| 推理延迟 | ms/query | ~200ms | **~20ms** (10×) | 辅助贡献四 |

---

## 3. 实验三：消融实验

### 3.1 目标

逐一去除系统的关键组件，量化每个组件对性能的独立贡献。

### 3.2 单组件消融

| 消融组 | 去掉的组件 | 替代方案 | 预期影响的指标 | 验证的贡献 |
|--------|-----------|----------|---------------|-----------|
| **C - LTM** | 去掉长期记忆 | 仅保留短期记忆 (最近 N 次交互) | 周期性购买 HR@10 ↓↓ (降幅 10-20%) | 核心贡献一 |
| **C - RL** | 去掉 RL 探索 | 改为确定性策略 (greedy / 固定规则) | 长期多样性 ↓↓ (降幅 15-25%), 长期留存 ↓ | 核心贡献二 |
| **C - UNC** | 去掉不确定性自适应 | 固定 ε=0.1 | 稀疏用户 HR@10 ↓ (降幅 5-10%), 探索效率 ↓ | 辅助贡献三 |
| **C - NFB** | 去掉负反馈信号 | 仅用正反馈 (与 P5 一致) | Reward 噪声 ↑, 多样性 ↓ (降幅 5-10%) | 负反馈建模价值 |
| **C - TIME** | 去掉时间衰减权重 | 均匀权重记忆检索 | 周期性模式识别 ↓ (降幅 5-10%) | 时间感知记忆价值 |
| **C - HIER** | 去掉层次化记忆 | 单一扁平记忆 (平替为 MR.Rec 风格的纯文本 RAG) | 检索精度 ↓, 长期依赖捕捉 ↓ | 两级记忆结构价值 |

### 3.3 2×2 因子设计 (核心机制交叉验证)

| | 无长期记忆 | 有长期记忆 |
|---|---|---|
| **无 RL 探索** | 纯 P5 + 短记忆 (近似 P5 baseline) | P5 + 长记忆 + 贪心决策 |
| **有 RL 探索** | P5 + RL + 短记忆 | **完整系统 (C)** |

该设计可同时观察长记忆与 RL 的主效应及交互效应，交互效应反映了"记忆驱动的自适应探索"是否产生 1+1>2 的增益。

### 3.4 行动掩码消融

| 消融组 | 配置 | 预期影响 |
|--------|------|----------|
| **C - MASK** | 去掉动作掩码 (已购/过敏品类过滤) | 无效推荐率 ↑, 用户体验 ↓ |
| **C - MASK_PARTIAL** | 保留动作掩码但去掉物品级过滤 | 中间效果 |

验证确定性安全检查 (action masking) 对推荐质量的影响。

---

## 3.5 精细化多样性/探索收益实验 (针对 Gini ≈ 0.45 的数据特性)

### 3.5.1 问题分析

数据探索显示 Gini 系数 ~0.45-0.50（头部 20% 物品占 52-58% 交互），长尾分布不如社交/短视频场景极端。这意味着 RL 探索的多样性增益需要更精细的指标设计才能被有效度量。单纯看 "全量 Diversity@k" 容易被头部物品的稳定表现淹没。本节设计 5 个子实验，从不同维度拆解和量化探索收益。

### 3.5.2 子实验 A：流行度分层 Recall 分析 (Popularity-Stratified HR@k)

**核心思路**：将物品按训练集中出现频次分成 5 个等宽桶 (each 20% quantile)，分别报告 HR@10 和 NDCG@10。

| 物品桶 | 定义 (频次分位) | 代表含义 | P5 预期 | 本系统预期 |
|--------|---------------|----------|---------|-----------|
| **Head** | 80-100% | 热门爆款 | HR@10 高 | HR@10 持平或略低 |
| **Mid-Head** | 60-80% | 较热门 | HR@10 中 | HR@10 持平 |
| **Mid-Tail** | 40-60% | 腰部物品 | HR@10 中低 | HR@10 **略高** |
| **Tail** | 20-40% | 长尾 | HR@10 低 | HR@10 **显著高** |
| **Cold** | 0-20% | 冷门/新物品 | HR@10 ≈ random baseline | HR@10 **显著高** |

**衡量方式**：

```
Popularity-Stratified Gain (PSG) = Σ_{bucket} w_i · (HR_ours(i) - HR_p5(i)) / HR_p5(i)
```

其中 w_i 为桶 i 的逆流行度权重 `w_i = 1 / popularity_i`，使 Cold/Tail 桶的改善获得更高权重。

**关键统计检验**：
- 对每个桶分别做配对 t-test (Bonferroni 校正)
- 重点看 Cold 桶的 effect size (Cohen's d)，预期 d > 0.5
- 同时报告 Random baseline 在各桶的 HR@10（证明 P5 在 Cold 桶是否已经失效）

**展示方式**：分组柱状图 + 连接线，横轴 5 个桶，纵轴 HR@10，三条柱 (Random / P5 / Ours)。

### 3.5.3 子实验 B：回声室破除实验 (Echo Chamber Breaking)

**核心思路**：在真实数据基础上构造 "回声室用户子集"，验证 RL 探索能否主动打破信息茧房。

**回声室用户定义**：过去 N 次交互中，≥80% 的物品集中在 ≤3 个细粒度类别中（如 Beauty → Hair Care → Shampoo）。

**实验协议**：

1. 在 Beauty 测试集中筛选符合回声室定义的保守型用户（预计 15-25%）
2. 仅使用前一半交互历史作为输入，预测后一半交互的物品
3. 对预测结果计算两个关键指标：
   - **Category Expansion Rate (CER)**：推荐列表中来自用户历史高频类别之外的物品占比
   - **Non-Dominant HR@k**：仅对用户非主要类别物品计算的 Hit Rate
4. 对比 P5 vs 本系统的 CER 和 Non-Dominant HR@k

| 指标 | 定义 | P5 预期 | 本系统预期 | 说明 |
|------|------|---------|-----------|------|
| CER | `#items_not_in_top3_categories / k` | <10% | **15-30%** | 打破茧房的意愿 |
| Non-Dominant HR@10 | 非主类别的 HR@10 | 低 | **显著高** | 探索质量 (不光敢推，还推得对) |
| Dominant HR@10 | 主类别的 HR@10 | 高 | 持平/略低 | 核心能力不丢失 |
| CER-to-Hit Ratio | Non-Dominant HR / CER | N/A | **越高越好** | 探索效率 (敢推+推准) |

**关键洞察**：如果 CER 高但 Non-Dominant HR 低，说明在乱探索；如果 CER 和 Non-Dominant HR 都高，说明探索是有方向性的——这正是不确定性+记忆驱动的自适应探索的价值。

### 3.5.4 子实验 C：用户类别分布漂移追踪 (Category Drift Tracking)

**核心思路**：在仿真环境中逐日追踪用户交互物品的类别分布，对比 P5 与本系统下的分布变化。

**协议**：
- 初始化：对每个用户计算 Day 0 的类别分布熵 H_0 = -Σ p(c) log p(c)
- 仿真 30 天，每天记录当日推荐的 20 个物品的类别分布熵 H_t
- 计算 ΔH = H_30 - H_0

| 用户类型 | P5 ΔH 预期 | 本系统 ΔH 预期 | 解读 |
|----------|-----------|---------------|------|
| 保守型 | **负** (类别收窄，茧房加剧) | **正** (温和扩展) | 打破茧房 |
| 探索型 | 略负 | **正** | 满足探索欲 |
| 规律购买型 | ~0 | ~0 | 均尊重周期性需求 |
| 稀疏型 | N/A (高噪声) | **正** (不确定性驱动) | 冷启动探索 |

**展示方式**：4 个子图 (按用户类型)，x 轴时间，y 轴 H_t，两条曲线 (P5 / Ours)，阴影为 ±1 std。

### 3.5.5 子实验 D：探索-利用 Pareto 前沿分析

**核心思路**：通过变化 RL 的探索强度 ε ∈ {0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5}，绘制 Diversity@20 vs HR@10 的 Pareto 曲线，展示本系统的灵活性和最优工作点。

**对比基线**：
- P5 的固定点（单一结果，无法调节）
- P5 + temperature sampling（变化 T ∈ {0.5, 0.7, 1.0, 1.2, 1.5} 调节多样性）

| 曲线 | 调节方式 | 预期 |
|------|----------|------|
| P5 + Temperature | 增大 T → 多样性上升，但 HR 急剧下降（不可控随机） | Pareto-inefficient |
| 本系统 + ε | 增大 ε → 多样性上升，HR 温和下降（有方向性探索） | **Pareto-efficient** |

**核心论证**：在相同多样性水平下，本系统的 HR@10 始终高于 P5+Temperature。这是因为 RL 探索是 "语义驱动的"（动作空间 `explore_new_category` 有明确语义目标），而非仅靠 softmax 温度随机抖动。

### 3.5.6 子实验 E：冷物品发现与孵化 (Cold-Item Discovery & Incubation)

**核心思路**：跟踪训练集中交互数 ≤ 5 的 "冷物品"，看本系统是否能将它们 "孵化" 为活跃物品。

**协议**：
1. 在训练集中标注所有交互数 ≤ 5 的 cold items
2. 在测试/仿真阶段，跟踪这些 cold items 被推荐的次数和点击率
3. 定义两个核心指标：

| 指标 | 定义 | P5 预期 | 本系统预期 |
|------|------|---------|-----------|
| **Cold-Item Coverage (CIC)** | 至少被推荐过 1 次的 cold-item 数 / 总 cold-item 数 | <5% | **15-30%** |
| **Cold-Item Hit Rate (CI-HR)** | cold-item 被推荐且被点击的比例 | ≈ 0% | **5-12%** |
| **Incubation Rate** | 仿真后 cold-item 交互数 > 5 的比例 | ≈ 0% | **3-8%** |

**展示方式**：Venn 图或 Upset plot，展示 P5 推荐物品集 vs 本系统推荐物品集中 Cold item 的交叠情况。

### 3.5.7 数据洞察 → 实验设计映射

| 数据发现 | 对应子实验 | 预期验证的贡献 |
|----------|-----------|---------------|
| 头部 20% 物品占 52-58% 交互 | A: 分层 Recall | 探索在长尾物品上的精度提升 |
| 保守型用户 (32% ≤5 交互) 可能陷入窄兴趣 | B: 回声室破除 | RL 主动探索打破茧房 |
| 类别多样性充足 (Beauty 656 类) | C: 类别漂移 | 记忆 + 探索维持类别多样性 |
| Gini ~0.45, 中等长尾 | D: Pareto 前沿 | 探索可控 (非随机抖动) |
| 12K 物品中大量冷物品 | E: 冷物品发现 | 探索的商业价值 (孵化新品) |

---

## 4. 实验四：推理效率对比

### 4.1 目标

量化论证离散动作 RL 策略相对于 Beam Search 的推理效率优势。

### 4.2 测试配置

| 配置项 | P5 (Beam=20) | 本系统 (RL) | 本系统 (RL + LLM解释) |
|--------|-------------|-----------|---------------------|
| Decoder 前向次数 | 20 | 1 | 1-2 |
| 记忆检索次数 | 0 | 1 (<10ms) | 1 (<10ms) |
| 候选排序 | O(B × V) | N/A (动作预定义) | N/A |
| 自然语言生成 | 可选 | 无 | 可选 (贪心解码, 1次前向) |

### 4.3 测试指标

| 指标 | 测量方法 |
|------|----------|
| 单次推理延迟 (ms) | 测量 p50/p95/p99, batch=1 |
| Throughput (queries/sec) | 测量 batch=1, 4, 8, 16, 32 |
| GPU 显存占用 (GB) | torch.cuda.max_memory_allocated |
| 端到端推荐延迟 (含记忆检索) | 包含 ChromaDB/FAISS 检索耗时 |

### 4.4 硬件

- 单卡测试: NVIDIA A100 80GB / RTX 4090 24GB
- Batch size 1 和 batch size 32 两组

### 4.5 预期结论

本系统每次推荐的 decoder 调用数从 20 降至 1，推理延迟降低约一个数量级，适合需要低延迟的实时推荐场景。

---

## 5. 实验五：长期收益对比 (仿真环境) ⭐ 关键实验

### 5.1 目标

在用户模拟环境中对比 P5 与本系统在 30 天长期交互下的累积奖励、留存率、多样性差异。核心目标是展示两条反直觉曲线：**(1) 短期 CTR 略降但长期反超；(2) 类别熵 P5 持续收窄而本系统维持或扩张。** 这是论文最有区分度的 insight。

### 5.2 用户模拟器设计 (数据校准版)

#### 5.2.1 用户画像校准 (基于 data_exploration.py 的真实统计)

| 用户属性 | 真实数据统计 (Beauty) | 模拟器设置 |
|----------|---------------------|-----------|
| 总用户数 | 22,363 | 采样 2,000 |
| 平均序列长度 | 8.9 | 历史 warm-up: 7 天 ≈ 均值 |
| 稀疏用户 (≤5) 占比 | 32.0% | 仿真中采样 30% |
| 密集用户 (>50) 占比 | 0.7% | 仿真中采样 5% (过采样以观察长期效应) |
| 规律购买用户 (CV<0.5) | 29.2% | 仿真中采样 25% |
| 负反馈 (rating≤2) 占比 | 37.2% | 仿真中 P(skip)=35% |
| 类别总数 | 656 (Beauty) | 仿真中简化为 50 大类 |

#### 5.2.2 仿真环境架构

```
┌──────────────────────────────────────────────────┐
│                   User Simulator                  │
│                                                  │
│  State (per user):                               │
│    - 兴趣向量 u_t ∈ R^50 (50维类目偏好)           │
│    - 满意度 s_t ∈ [0,1] (EMA平滑)                 │
│    - 疲劳度 f_t(c) per category (重复同类累积)    │
│    - 周期需求时钟 clock_c per category            │
│    - 物品交互历史 H_t                             │
│                                                  │
│  动力学:                                         │
│    u_{t+1} = u_t + η*(item_emb - u_t)*click      │
│            + ε_drift * N(0,1)                    │
│    s_t = EMA(click_success, α=0.9)               │
│    f_t(c) = f_{t-1}(c) * 0.9 + 0.1*I(item∈c)   │
│    clock_c decrements daily, resets on purchase   │
│                                                  │
│  Behavior Model:                                 │
│    score(item) = α·sim(u, emb_item)              │
│                 + β·category_novelty             │
│                 - γ·fatigue(item_category)        │
│                 + δ·periodic_boost(clock_c ≤ 0)  │
│    P(click) = σ(score)                           │
│    P(skip)  = σ(θ_skip - score)·neg_ratio        │
│    P(stay)  = σ(η·s_t + ζ·diversity_recent - ω) │
└──────────────────────────────────────────────────┘
```

#### 5.2.3 模拟器参数 (数据校准)

| 参数 | 含义 | 取值 | 校准依据 |
|------|------|------|----------|
| `α` | 兴趣匹配权重 | 1.0 | — |
| `β` | 类别新颖性偏好 | N(0.25, 0.12) | 个体差异 — 29%规律用户 β 为负 |
| `γ` | 同类别疲劳 | 0.15 | 同类连续推荐 5 次后 click prob 降 ~50% |
| `δ` | 周期性需求 boost | 0.4 | 仅对规律购买型用户的特定类别生效 |
| `η` | 留存-满意度系数 | 0.8 | — |
| `ζ` | 留存-多样性系数 | 0.2 | 多样性对留存的独立贡献 |
| `ω` | 留存阈值 | 0.35 | 使得总体留存率 ≈ 60-70% |
| `θ_skip` | 跳过阈值 | -0.5 | 使得 skip rate ≈ 35% (校准到负反馈占比) |
| 兴趣漂移噪声 | ε_drift | 0.03/月 | 缓慢漂移 |
| 疲劳衰减速率 | λ_fatigue | 0.9/day | 约 10 天半衰期 |

#### 5.2.4 用户分群与采样配比

| 用户类型 | 占比 | α | β | 周期性时钟 | 关键测试目标 |
|----------|------|----|-----|-----------|-------------|
| 规律购买型 | 25% | 1.0 | -0.1 | 90±30d per category | 长期记忆 + 周期预测 |
| 保守型 (茧房风险) | 30% | 1.2 | -0.3 | 无 | 回声室破除 (子实验 B) |
| 探索型 | 20% | 0.8 | 0.5 | 无 | 多样性受益 |
| 稀疏型 (冷启动) | 15% | 0.6 | 0.4 | 无 | 不确定性自适应探索 |
| 混合型 | 10% | 1.0 | 0.15 | 部分类别有周期 | 综合评估 |

### 5.3 仿真协议

| 参数 | 取值 | 说明 |
|------|------|------|
| 仿真天数 | 30 天 | Day 1-7 warm-up, Day 8-30 正式评估 |
| 每日推荐数 | 20 个物品 | 模拟一次推荐 session |
| 用户数 | 2,000 | 按 5.2.4 配比采样 (每组 ≥300) |
| 物品池 | 12,101 (Beauty 全集) | 与真实数据一致 |
| 物品流行度分布 | 拟合真实 Gini ≈ 0.49 | 用真实 `sequential_data.txt` 统计 |
| 物品类别分布 | 拟合真实 656 类 | 用真实 `meta.json.gz` categories |
| 随机种子 | 5 个 (42, 123, 456, 789, 1024) | 结果取 mean ± std |
| 对比系统 | P5-small / 本系统 (C) / C-RL / C-LTM | 四组并跑 |

### 5.4 评估指标 (仿真环境 + 真实数据)

#### 5.4.1 主指标 (Table 必报)

| 指标 | 符号 | 定义 | 预期 (P5 vs Ours) |
|------|------|------|-------------------|
| 30-Day Cumulative Reward | CR_30 | Σ_{t=1}^{30} R_t | Ours > P5 (↑10-20%) |
| Day 30 Retention | RET_30 | #users_active_day30 / N | Ours > P5 (↑5-10%) |
| Mean Diversity@20 | DIV | 1/N Σ_t (1 - avg_cos_sim(top20)) | Ours >> P5 (↑15-25%) |
| Category Entropy (Day 30) | H_30 | -Σ_c p_30(c) log p_30(c) | Ours > P5 |
| Periodic Purchase Recall | PPR | HR@10 for items due per clock | Ours >> P5 (贡献一) |
| CTR (Day 1-7 vs Day 24-30) | CTR_early / CTR_late | 早期 vs 晚期对比 | P5早期高, Ours晚期反超 |

#### 5.4.2 辅助指标 (Section 3.5 协同)

| 指标 | 来源 | 说明 |
|------|------|------|
| Popularity-Stratified HR@10 | 子实验 A | 分桶在仿真中的动态变化 |
| CER (Category Expansion Rate) | 子实验 B | 茧房用户的类别扩展率 |
| Category Entropy ΔH (Day 30 - Day 0) | 子实验 C | 按用户类型分组的漂移量 |
| Cold-Item Coverage | 子实验 E | 冷物品被推荐次数 |

#### 5.4.3 反直觉指标 (拟作为 Fig.1/Teaser)

| 指标 | 故事 |
|------|------|
| **Cross-over Day** | CTR 曲线交叉的时间点 (Bootstrap CI) |
| **Diversity Deficit Reduction** | (H_optimal - H_P5) / (H_optimal - H_ours) — 多样性的 "修复比例" |
| **Exploration ROI** | (CR_ours - CR_P5) / (#explore_actions_taken) — 每次探索的长期收益 |

### 5.5 预期结果 (更新)

#### 5.5.1 主结果表 (Table: Simulation Results)

```
                               P5         Ours (C)    C-RL        C-LTM
─────────────────────────────────────────────────────────────────────────
CR_30 (Cumulative Reward)      1.00×      1.12-1.18×  1.03-1.06×  1.05-1.10×
RET_30 (Retention)             58-62%     65-72%       60-63%      62-67%
DIV (Diversity@20)             0.15-0.25  0.35-0.45    0.18-0.28   0.22-0.32
H_30 (Category Entropy)        1.8-2.2    2.8-3.4      2.0-2.5     2.3-2.8
PPR (Periodic Purchase Rec.)   0.05-0.10  0.18-0.25    0.06-0.12   0.12-0.18
Cross-over Day (CTR)           N/A        Day 12-18    Day 22+     Day 15-20
Cold-Item Coverage             3-5%       18-25%       6-10%       10-15%
CER (Echo Chamber Users)       5-10%      22-30%       12-18%      10-15%
Exploration ROI                N/A        0.35-0.50    0.10-0.20   0.15-0.25
```

#### 5.5.2 关键曲线

```
CTR (按天)
^
│     P5 ████████╲
│                 ╲__________  (单调衰减, 疲劳)
│     Ours ___╱╱╱╱
│             ╱‾‾‾‾‾‾‾‾‾‾‾  (先低后高, Day 12-18 反超)
│
└──────────────────────────────> Day
     0    7    15         30
     ├Warmup┤← 正式评估期 →┤


Diversity@20 (按天)
^
│     Ours ──────∼∼∼∼────── (高位波动)
│     P5    ───╲___________  (缓慢衰减)
│
└──────────────────────────────> Day


Category Entropy (按用户类型, Day 30 - Day 0)
     ΔH
     ↑
    +1│     ●探索型(Ours)
      │
      │  ●保守型(Ours)     ●规律型(Ours)
      │
     0├──────────────────────────── P5 全类型
      │                      ●保守型(P5)
    -1│                            
      └──────────────────────────> 用户类型

Cold-Item Coverage (累计, 按天)
     ↑
     │     Ours ╱╱╱╱╱╱
     │        ╱
     │   P5 ╱
     └──────────────────────────> Day (P5 几乎不增长)
```

### 5.6 统计检验与鲁棒性

#### 5.6.1 主检验

| 检验 | 方法 | 阈值 |
|------|------|------|
| CR_30 差异 | 配对 t-test (5 seeds, N=2000 pairs) | p < 0.01 |
| Cross-over Day | Bootstrap (n=1000), 报告 95% CI | CI 不包含 Day 30 |
| ΔH per user type | 2-way ANOVA (system × user_type) | p < 0.05, 含交互项 |
| Cold-Item Coverage | Fisher's exact test (被推荐/未被推荐) | p < 0.001 |

#### 5.6.2 敏感性分析

| 分析 | 方法 | 目的 |
|------|------|------|
| 参数敏感性 | 对 β, γ, δ 分别做 ±30% 扰动 | 验证结论对模拟器参数不敏感 |
| 用户配比敏感性 | 改变 5 类用户比例 ±50% | 验证结论对采样偏差鲁棒 |
| 物品池大小敏感性 | 测试 5K / 10K / 全量 12K | 规模鲁棒性 |
| 天数敏感性 | 仿真 30/45/60 天 | 验证长期趋势是否持续 |

#### 5.6.3 真实数据回放验证 (Off-Policy Evaluation)

仿真环境的核心风险是用户模型的真实性。为增强说服力，额外做真实数据 off-policy 评估：

1. 使用 Beauty 训练集训练 P5 和本系统
2. 在测试集上用 **Inverse Propensity Scoring (IPS)** 估计两系统的期望 reward
3. 使用 **Doubly Robust (DR)** 估计器作为交叉验证
4. 如果 OPE 估计的方向与仿真一致（Ours > P5），则仿真结论获得真实数据支持

| 方法 | 说明 |
|------|------|
| IPS | 基于 logging policy (P5) 的 propensity score 做 off-policy 纠正 |
| DR | IPS + 回归模型双重鲁棒 |
| CAPE | 针对推荐列表 (slate) 的 off-policy 估计 |

---

## 6. 实验六：补充分析实验

### 6.1 记忆检索质量分析

| 实验 | 方法 | 指标 |
|------|------|------|
| 检索相关性 | 给定用户当前意图，评估 top-k 记忆片段的实际相关率 | Recall@k, Precision@k, MRR |
| 对比基线 | 随机检索 vs TF-IDF vs 稠密向量检索 | 同上 |
| 时间衰减敏感度 | 调整衰减系数 α∈{0.01, 0.1, 0.5, 0.9, 0.99} | 周期性购买的 HR@10 |
| 压缩维度敏感度 | 记忆向量维度 d∈{64, 128, 256, 512, 768} | 检索速度 vs 精度 trade-off |

### 6.2 动作分布分析

| 分析 | 说明 |
|------|------|
| 按用户类型统计动作分布 | 新用户 vs 老用户, 稀疏 vs 密集 |
| 动作切换频率 | exploit→explore 转换率 |
| 探索动作的后续收益 | 执行 `explore_new_category` 后 7 天内的用户行为变化 |

### 6.3 参数增量公平性分析

| 组 | 参数量 | 说明 |
|----|--------|------|
| P5-small | 60.75M | 基线 |
| P5-small + 记忆编码器 | 60.75M + X M | 仅加记忆 |
| P5-small + RL 策略 | 60.75M + Y M | 仅加 RL |
| 完整系统 | 60.75M + X + Y M | 全部新增模块 |
| P5-base | 223.28M | 更大参数基线 |

若 P5-base (223M) 在长周期任务上仍不及完整系统 (60.75M + 少量新增参数)，则有力证明增益来自架构设计而非参数堆叠。

### 6.4 负反馈信号消融 (细化)

| 负反馈类型 | 信号来源 | 加入方式 |
|-----------|----------|----------|
| 快速跳过 | 停留时间 < 2s | Reward -= 0.1 |
| 退货 | 购买后退货 | Reward -= 0.5 |
| 差评 | rating ≤ 2 | Reward -= 0.3 |
| 显式不感兴趣 | "not interested" 标记 | Reward -= 0.2 |

每类负反馈单独 + 组合的消融实验，分析哪种负反馈对 reward 质量贡献最大。

### 6.5 长序列用户子集分析

从数据集中筛选出交互长度 > 50 的长序列用户 (~20% 用户)，单独对比 P5 和本系统在这类用户上的性能差异，预期差距更大。

---

## 7. 实验执行时间线

| 阶段 | 实验 | 预计时间 | 里程碑 |
|------|------|---------|--------|
| 第 1 周 | 实验一：P5 复现 + 数据探索 | 1 周 | P5 复现结果与论文一致 |
| 第 2-3 周 | 记忆系统 MVP + 离线 RL 训练 | 2 周 | 可运行的基础系统 |
| 第 4 周 | 实验二：主对比实验 | 1 周 | 核心结果表格 |
| 第 5 周 | 实验三：消融实验 (含 3.4 精细化探索) | 1 周 | 消融矩阵 + 分层 recall 曲线 |
| 第 6 周 | 实验四 + 子实验 B,C,E 的真实数据部分 | 1 周 | 效率图表 + 回声室/冷物品真实数据结果 |
| 第 7-9 周 | 实验五：仿真环境 + 全部分析 + OPE 验证 | 3 周 | **核心 insight 验证 + 真实数据交叉验证** |
| 第 10-14 周 | 论文撰写 | 4 周 | 投稿目标会议 |

---

## 8. 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| P5 复现结果与论文偏差 > 2% | 中 | 基准线不可信 | 联系原作者确认细节 / 使用 OpenP5 开源库交叉验证 |
| 记忆检索延迟高于预期 | 低 | 效率优势减弱 | 使用 FAISS GPU 加速 / 减小记忆维度 / 增加缓存 |
| 仿真环境的用户模型不真实 | 高 | 实验五结论不可信 | **三重交叉验证**: (1) 参数敏感性分析 (±30% 扰动); (2) 用户配比鲁棒性测试; (3) 真实数据 OPE (IPS/DR) |
| RL 训练不稳定 | 中 | 主实验无法完成 | 先用行为克隆 (BC) 初始化策略，再用离线 RL 微调 |
| 完整系统在短期任务上显著低于 P5 | 低 | 论文 story 受损 | 分析原因 (过探索？记忆噪声？)，调整 reward 权重 |
| 新增参数量过大导致不公平对比 | 中 | 审稿人质疑 | 诚实报告参数增量，设参数量消融组 (Section 6.3) |
| **多样性增益在 Gini ≈ 0.45 数据上不显著** | **中高** | **核心贡献二缺乏实验支撑** | **(1) 分层指标 (子实验 A) 将增益定位到 Cold/Tail 桶; (2) 回声室用户子集 (子实验 B) 放大效应; (3) Pareto 前沿 (子实验 D) 展示效率优势而非绝对数值; (4) 如果全量 Diversity@20 确实无差异，诚实报告并转向子集分析** |
| Cold 桶物品在真实测试中无法被评估 (交互太少) | 中 | 子实验 A/E 结论不可靠 | 仅在仿真中做 Cold 物品分析; 真实数据中合并 Cold+Tail 桶确保评估样本量 |
| P5+Temperature 基线表现超预期 | 低 | Pareto 前沿故事受损 | 同时报告 temperature 采样带来的 BLEU/ROUGE 退化 (文本生成质量)，论证 RL 探索不影响生成质量 |
| Cross-over Day 在 30 天内未出现 | 低中 | 核心 teaser 失效 | 延长仿真至 60 天; 或改为 report "Day 30 CTR gap reduction" |

---

## 9. 投稿目标

| 会议 | 截稿日期 (预估) | 说明 |
|------|---------------|------|
| RecSys 2027 | ~2027 年 5-6 月 | 首选，推荐系统顶会 |
| CIKM 2027 | ~2027 年 3-4 月 | 备选，信息检索/数据挖掘 |
| SIGIR 2027 | ~2027 年 1-2 月 | 信息检索顶会 |
