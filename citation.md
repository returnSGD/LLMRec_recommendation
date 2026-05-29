# 相关论文引用清单

> 本文档整理与 idea.md 中"RL + 长时序记忆推荐系统"相关的论文，按主题分组。
> 每篇论文标注了与 idea 的相似度（⭐越多越相似）和关键差异。

---

## 1. P5 Baseline 及其后续工作

### 1.1 P5: Recommendation as Language Processing (RLP)
- **作者**: Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang (Rutgers)
- **发表**: ACM RecSys 2022
- **arXiv**: [2203.13366](https://arxiv.org/abs/2203.13366)
- **代码**: [github.com/jeykigung/P5](https://github.com/jeykigung/P5)
- **引用量**: ~660+
- **核心思路**: 将所有推荐任务统一为 text-to-text 框架，用个性化 prompt 覆盖评分预测、序列推荐、解释生成等 5 类任务，支持 zero-shot/few-shot 迁移。
- **与 idea 的关系**: **直接 baseline**，你的 idea 要超越的对象。P5 无长期记忆、无 RL 决策、依赖固定上下文窗口。
- **文件**: `papers/P5_RLP_2203.13366.pdf`

### 1.2 PAP-REC: Personalized Automatic Prompt for Recommendation Language Model
- **作者**: Zelong Li, Jianchao Ji, Yingqiang Ge, Wenyue Hua, Yongfeng Zhang (Rutgers)
- **发表**: ACM TOIS 2024
- **arXiv**: [2402.00284](https://arxiv.org/abs/2402.00284)
- **代码**: [github.com/rutgerswiselab/PAP-REC](https://github.com/rutgerswiselab/PAP-REC)
- **核心思路**: 用梯度优化自动生成个性化 prompt tokens（替代手工 prompt），不同用户不同 prompt。
- **与 idea 的关系**: P5 的 prompt 工程改进，不涉及记忆或 RL。
- **文件**: `papers/PAP-REC_2402.00284.pdf`

### 1.3 UP5: Unbiased Foundation Model for Fairness-aware Recommendation
- **作者**: Wenyue Hua, Yingqiang Ge, Shuyuan Xu, Jianchao Ji, Zelong Li, Yongfeng Zhang (Rutgers)
- **发表**: EACL 2024 (Long Paper)
- **arXiv**: [2305.12090](https://arxiv.org/abs/2305.12090)
- **代码**: [github.com/agiresearch/UP5](https://github.com/agiresearch/UP5)
- **核心思路**: 在 P5 基础上引入 Counterfactually-Fair-Prompting (CFP)，通过对抗训练 fairness prefix prompt 消除敏感属性偏差，避免全模型重训练。
- **与 idea 的关系**: P5 的公平性扩展，不涉及记忆或 RL 决策。
- **文件**: `papers/UP5_2305.12090.pdf`

### 1.4 IDGenRec: LLM-RecSys Alignment with Textual ID Learning
- **作者**: Juntao Tan, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Zelong Li, Yongfeng Zhang (Rutgers)
- **发表**: SIGIR 2024
- **arXiv**: [2403.19021](https://arxiv.org/abs/2403.19021)
- **代码**: [github.com/agiresearch/IDGenRec](https://github.com/agiresearch/IDGenRec)
- **核心思路**: 为每个 item 生成唯一、简洁、语义丰富、平台无关的 textual ID，使 LLM-based 推荐模型可跨数据集迁移。在 19 个数据集上训练、6 个 unseen 数据集上 zero-shot 测试。
- **与 idea 的关系**: 解决 P5 中 item ID 的语义表示问题，可作为你底层渲染模块的 item 表示方案。
- **文件**: `papers/IDGenRec_2403.19021.pdf`

---

## 2. RL + 记忆/检索增强推荐（最直接竞品）⭐⭐⭐⭐⭐

### 2.1 MR.Rec: Synergizing Memory and Reasoning for Personalized Recommendation Assistant with LLMs
- **作者**: Jiani Huang et al.
- **发表**: arXiv preprint, 2025.10
- **arXiv**: [2510.14629](https://arxiv.org/abs/2510.14629)
- **相似度**: ⭐⭐⭐⭐⭐
- **核心思路**: **RL + RAG + 记忆 + 推荐四合一**。构建 RAG 系统索引外部记忆，设计 reasoning-enhanced memory retrieval（超越简单 query matching），用 RL 训练 LLM 自主学习记忆利用和推理策略。
- **关键差异**: 记忆仍是文本/RAG 形式，无向量压缩+时间衰减机制；无行为树安全层；使用普通 MDP 而非 POMDP；记忆检索依赖 LLM 推理而非向量相似度。
- **与你的 idea 的重叠**: 最高！同属 "RL + 外部记忆 + 推荐" 范式。需要重点区分的竞品。
- **文件**: `papers/MR.Rec_2510.14629.pdf`

### 2.2 CoRAL: Collaborative Retrieval-Augmented Large Language Models Improve Long-tail Recommendation
- **作者**: Junda Wu, Cheng-Chun Chang, Tong Yu, Zhankui He, Jianing Wang, Yupeng Hou, Julian McAuley
- **发表**: 2024.03
- **arXiv**: [2403.06447](https://arxiv.org/abs/2403.06447)
- **相似度**: ⭐⭐⭐⭐
- **核心思路**: 将协同过滤信号注入 RAG 检索，用 RL 学习检索策略（从交互历史中选取最小充分信息子集），专注长尾推荐。
- **关键差异**: 记忆是协同过滤交互信号，非用户行为记忆；检索目标是最小充分信息集，非时效性记忆；无行为树、无 POMDP。
- **与你的 idea 的重叠**: RL 学习检索策略的思路相似，但记忆内容和目标不同。
- **文件**: `papers/CoRAL_2403.06447.pdf`

### 2.3 RALLRec: Improving Retrieval Augmented Large Language Model Recommendation with Representation Learning
- **作者**: Jian Xu, Sichun Luo, Xiangyu Chen, Haoming Huang, Linqi Song
- **发表**: TheWebConf 2025 (WWW'25 Short Paper)
- **arXiv**: [2502.06101](https://arxiv.org/abs/2502.06101)
- **代码**: [github.com/JianXu95/RALLRec](https://github.com/JianXu95/RALLRec)
- **相似度**: ⭐⭐⭐
- **核心思路**: 用 LLM 生成更详细的 item 描述 + 联合表示学习（文本语义 + 协同信号）+ 重排序方法捕捉时序偏好动态。
- **关键差异**: 无 RL；检索优化靠表示学习而非策略学习；记忆范围限定在当前推荐上下文的 item-item 关系。
- **与你的 idea 的重叠**: RAG for RecSys 思路相似，但聚焦于 item 表示而非用户长期记忆。
- **文件**: `papers/RALLRec_2502.06101.pdf`

### 2.4 EMG-RAG: Crafting Personalized Agents through Retrieval-Augmented Generation on Editable Memory Graphs
- **作者**: Wang, Li et al.
- **发表**: EMNLP 2024
- **arXiv**: [2409.19401](https://arxiv.org/abs/2409.19401)
- **相似度**: ⭐⭐⭐
- **核心思路**: 用可编辑记忆图 (EMG) 存储用户记忆，RL 智能体（MDP + Policy Gradient/REINFORCE）导航图结构进行检索。已部署在真实智能手机 AI 助手中。
- **关键差异**: **非推荐场景**（通用 AI 助手）；记忆图结构 vs 你的向量库+时间衰减；无行为树安全层。
- **与你的 idea 的重叠**: RL + 记忆图检索的架构相似，但应用场景完全不同。
- **文件**: `papers/EMG-RAG_2409.19401.pdf`

### 2.5 ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation
- **作者**: Walmart Global Tech
- **发表**: SIGIR 2025
- **arXiv**: [2506.21931](https://arxiv.org/abs/2506.21931)
- **相似度**: ⭐⭐⭐
- **核心思路**: 4 个专业 LLM Agent（User Understanding, NLI, Context Summary, Item Ranker）+ Blackboard 共享记忆，有 memory moderation 方案处理长期+会话行为。NDCG@5 提升 42.1%。
- **关键差异**: 无 RL 训练（纯 Agentic 流水线，非策略优化）；无行为树；记忆管理是 moderation 而非向量检索。
- **与你的 idea 的重叠**: 多模块解耦架构 + 记忆管理思路相似，但采用 Agent 协作而非 RL 决策+行为树。
- **文件**: `papers/ARAG_2506.21931.pdf`

### 2.6 RecGPT-V2 Technical Report
- **作者**: 阿里巴巴/淘宝
- **发表**: 2025.12
- **arXiv**: [2512.14503](https://arxiv.org/abs/2512.14503)
- **相似度**: ⭐⭐½
- **核心思路**: 层次化多智能体系统 + 约束 RL 缓解多奖励冲突 + 混合表示推理压缩行为上下文（32K→11K tokens）+ Agent-as-a-Judge 评估。线上收益：CTR +2.98%, IPV +3.71%。
- **关键差异**: 工业级部署，记忆是行为上下文压缩而非外置向量库；RL 用于约束多奖励冲突而非长期记忆检索决策；无行为树安全层。
- **与你的 idea 的重叠**: 层次化架构 + 约束 RL 思路相似，但工程化导向，非学术创新。
- **文件**: `papers/RecGPT-V2_2512.14503.pdf`

---

## 3. 离线 RL 推荐（反馈循环/因果去偏）⭐⭐⭐⭐

### 3.1 MocDT: Future-Conditioned Recommendations with Multi-Objective Controllable Decision Transformer
- **作者**: Chongming Gao, Kexin Huang, Ziang Fei, Jiaju Chen, Jiawei Chen, Jianshan Sun, Shuchang Liu, Qingpeng Cai, Peng Jiang
- **发表**: CIKM 2024
- **arXiv**: [2501.07212](https://arxiv.org/abs/2501.07212)
- **相似度**: ⭐⭐⭐⭐
- **核心思路**: 用 Decision Transformer 架构生成 item 序列，通过 control signal（如 (1.0, 0.0) = 纯优化评分，(0.0, 1.0) = 纯优化多样性）在推理时切换优化目标，无需重训练。包含长期累积奖励和多样性目标。
- **关键差异**: 基于 DT（非传统 RL），条件生成而非主动探索；无外部记忆；目标是多目标 trade-off 而非打破回声室。
- **与你的 idea 的重叠**: 优化长期多样性目标、对抗短期 CTR 偏好的思路高度一致。是你需要重点引用的相关 work。
- **文件**: `papers/MocDT_2501.07212.pdf`

### 3.2 ROLeR: Effective Reward Shaping in Offline Reinforcement Learning for Recommender Systems
- **作者**: Yi Zhang, Ruihong Qiu, Jiajun Liu, Sen Wang (UQ, CSIRO)
- **发表**: CIKM 2024
- **arXiv**: [2407.13163](https://arxiv.org/abs/2407.13163)
- **相似度**: ⭐⭐⭐
- **核心思路**: 用 kNN 非参数奖励塑形修正离线稀疏数据中的奖励函数不准确性，提出 in-cluster distance-based uncertainty penalty，提供理论性能下界分析。SOTA on KuaiRec, KuaiRand, Coat, Yahoo。
- **关键差异**: 聚焦奖励函数工程，非完整架构创新；无记忆模块。
- **与你的 idea 的重叠**: 离线 RL 推荐中的奖励设计是你需要的技术组件，可引用其奖励塑形方法。
- **文件**: `papers/ROLeR_2407.13163.pdf`

### 3.3 PGCR: Policy-Guided Causal State Representation for Offline Reinforcement Learning Recommendation
- **作者**: Siyu Wang, Xiaocong Chen, Lina Yao (UNSW, CSIRO)
- **发表**: WWW 2025
- **arXiv**: [2502.02327](https://arxiv.org/abs/2502.02327)
- **相似度**: ⭐⭐⭐
- **核心思路**: 两阶段框架：先学习策略筛选因果相关状态成分 (CRCs)，用 Wasserstein 距离衡量因果效应；再训练编码器压缩状态表示。提供因果效应可辨识性的理论保证。
- **关键差异**: 聚焦状态表示学习，非整体系统架构；无记忆；无行为树。
- **与你的 idea 的重叠**: 因果状态建模的思想与你的 "MDP 解耦反馈循环" 高度相关，可作为你方法论部分的理论支撑引用。
- **文件**: `papers/PGCR_2502.02327.pdf`

### 3.4 CIDS: On Causally Disentangled State Representation Learning for RL-based Recommender Systems
- **作者**: Xiaocong Chen et al.
- **发表**: 2024.07
- **arXiv**: [2407.13091](https://arxiv.org/abs/2407.13091)
- **相似度**: ⭐⭐⭐
- **核心思路**: 将状态分解为 Directly Action-Influenced State Variables (DAIS) 和 Action-Influence Ancestors (AIA)，用条件互信息构建因果不可或缺的状态表示。PGCR 的前置工作。
- **关键差异**: 同 PGCR，聚焦状态表示。
- **与你的 idea 的重叠**: 因果解耦思想与你 MDP 中区分"真实偏好"和"展示偏差"的思路一致。
- **文件**: `papers/CIDS_2407.13091.pdf`

### 3.5 EDT4Rec: Maximum-Entropy Regularized Decision Transformer with Reward Relabelling for Dynamic Recommendation
- **作者**: Xiaocong Chen, Siyu Wang, Lina Yao (CSIRO, UNSW)
- **发表**: KDD 2024
- **arXiv**: [2406.00725](https://arxiv.org/abs/2406.00725)
- **相似度**: ⭐⭐½
- **核心思路**: 解决 Decision Transformer 两个缺陷：缺乏 trajectory stitching 能力和在线适应差。引入最大熵探索 + 奖励重标记。在 6 个离线数据集 + 在线模拟器验证。
- **关键差异**: 改进 DT 而非传统 RL；无记忆模块。
- **与你的 idea 的重叠**: 探索策略（最大熵）与你的 "主动探索打破茧房" 相关，可对比讨论。
- **文件**: `papers/EDT4Rec_2406.00725.pdf`

---

## 4. LLM Agent 架构 & 行为树安全层

### 4.1 Dendron: Behavior Trees Enable Structured Programming of Language Model Agents
- **作者**: Richard Kelley (University of Nevada)
- **发表**: 2024.04
- **arXiv**: [2404.07439](https://arxiv.org/abs/2404.07439)
- **相似度**: ⭐⭐（推荐系统场景下为 ⭐）
- **核心思路**: 提出 Dendron 库，用行为树结构化编程 LLM Agent。行为树叶节点 = 原子动作/条件，内部控制节点 (Sequence, Fallback) 提供确定性流控制。展示了一个 "未经过 RLHF 安全训练但仍满足安全约束" 的案例。
- **关键差异**: **应用于机器人和对话，非推荐系统**。推荐系统中使用行为树作为安全执行层目前是空白。
- **与你的 idea 的重叠**: 直接支撑你的创新点三（行为树安全执行层），是你架构中行为树部分必须引用的核心参考文献。
- **文件**: `papers/Dendron_BT_LLM_2404.07439.pdf`

---

## 5. 综述论文

### 5.1 A Survey on LLM-powered Agents for Recommender Systems
- **作者**: Qiyao Peng, Hongtao Liu, Hua Huang, Qing Yang, Minglai Shao
- **发表**: EMNLP 2025 Findings
- **arXiv**: [2502.10050](https://arxiv.org/abs/2502.10050)
- **核心思路**: 识别三种范式（Recommender-oriented, Interaction-oriented, Simulation-oriented），解剖四模块 Agent 架构（Profile → Memory → Planning → Action）。
- **与你的 idea 的关系**: 你论文 Related Work 部分必须引用的综述，支撑 "LLM Agent + RecSys" 方向的合法性。
- **文件**: `papers/LLM_Agents_RecSys_Survey_2502.10050.pdf`

### 5.2 Recommender Systems Meet Large Language Model Agents: A Survey
- **作者**: Xi Zhu, Yu Wang (Netflix), Hang Gao, Wujiang Xu et al.
- **发表**: Foundations and Trends in Privacy and Security, 2025
- **SSRN**: [5062105](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5062105)
- **GitHub**: [github.com/agiresearch/AgentRecSys](https://github.com/agiresearch/AgentRecSys)
- **核心思路**: 69 页全面综述 LLM Agent 与推荐系统的共生关系，涵盖 profile, memory, planning, action 组件及可信 AI (安全、可解释、公平、隐私)。
- **与你的 idea 的关系**: 同样适合 Related Work 引用，特别是 memory recommendation 和 tool recommendation 部分。
- **文件**: 未下载（SSRN 非 arXiv），建议通过机构访问或 GitHub 获取。

---

## 6. 你的 idea 中的 Baseline 方法

### 6.1 SASRec: Self-Attentive Sequential Recommendation
- **作者**: Wang-Cheng Kang, Julian McAuley (UCSD)
- **发表**: ICDM 2018
- **arXiv**: [1808.09781](https://arxiv.org/abs/1808.09781)
- **核心思路**: 用 self-attention 建模序列推荐，是序列推荐经典 baseline。
- **文件**: 需单独下载

### 6.2 S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization
- **作者**: Kun Zhou et al.
- **发表**: CIKM 2020
- **arXiv**: [2008.07873](https://arxiv.org/abs/2008.07873)
- **核心思路**: 用自监督学习增强序列推荐，通过互信息最大化学习 item-属性-序列之间的关联。
- **文件**: 需单独下载

---

## 7. 竞争格局总结

```
                    记忆机制
                        ↑
       MR.Rec ●        |        ● 你的 Idea
       (文本RAG)       |        (向量压缩+时间衰减
                        |         行为树安全层)
                        |
   CoRAL ●              |
   (CF信号检索)         |
                        |
  ─────────────────────────────────────→ RL 决策能力
                        |
      RALLRec ●         |         ● MocDT
      (无RL)            |         (DT条件生成)
                        |
                  ● ARAG
                  (Agent协作/无RL)
```

### 你的差异化优势（空白地带）

| 特征 | MR.Rec | CoRAL | MocDT | ARAG | **你的方案** |
|------|:------:|:-----:|:-----:|:----:|:---------:|
| RL 策略学习 | ✅ | ✅ | ❌(DT) | ❌ | ✅ |
| 外部长期记忆 | ✅(RAG文本) | ✅(CF信号) | ❌ | ✅(黑板书) | ✅(向量压缩+时间衰减) |
| 层次化记忆(短期+长期) | ❌ | ❌ | ❌ | 部分 | ✅ |
| POMDP 建模 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 行为树安全层 | ❌ | ❌ | ❌ | ❌ | ✅ |
| LLM 渲染层解耦 | 部分 | 部分 | N/A | ✅ | ✅ |
| 打破回声室/多样性探索 | ❌ | ❌ | ✅ | ❌ | ✅(核心motivation) |
| 三明治架构 | ❌ | ❌ | ❌ | 部分 | ✅ |

### 投稿建议

1. **最直接竞品 MR.Rec (2025.10)** 已发表 7 个月，需在 Introduction 中明确引用并区分。
2. **行为树安全层**在推荐系统中几乎是空白，可单独作为一个 contribution。
3. **"打破回声室"的 RL 目标**虽然 MocDT 涉及多样性优化，但你将其作为核心 motivation 而非 side objective 的做法仍然独特。
4. **建议投稿窗口**: RecSys 2026 (deadline 约 2026 年 5-6 月) 或 SIGIR 2027 (deadline 约 2027 年 1-2 月)。如果实验加速，可考虑 CIKM 2026 (deadline 约 2026 年 3-4 月)。
