这份 `README.md` 是专门为**第三问（影响因素分析）**设计的。请将此文档发送给负责写作的队友。

这份文档不仅解释了代码做了什么，更重要的是提供了**O奖级别的论文写作逻辑**，指导队友如何将图表转化为有深度的学术分析。

---

# 2026 MCM Problem C - Q3: 影响因素分析 (Factor Analysis)

## 📌 项目核心逻辑 (Core Logic)

在第三问中，我们需要回答：“是什么因素决定了选手的命运？”。为了达到 O 奖标准，我们不能只做简单的相关性分析，而是构建了一个**“双轨制归因模型” (Dual-Track Attribution Model)**。

我们融合了三份数据源：

1. **Q1 计算结果**：提供了“真实的”粉丝投票数据（Ground Truth）。
2. **官方原始数据**：提供了选手的基本信息。
3. **补充外部数据** (`dancing_with_the_stars_dataset.csv`)：补全了缺失的**年龄**和**行业**特征，极大增强了模型的鲁棒性。

我们使用了两个互补的模型：

* **模型 A (统计推断)**：线性混合效应模型 (LMM)。用于回答“**该因素的影响是否显著？(P-value)**”以及“**是正向还是负向影响？**”。
* **模型 B (机器学习)**：随机森林 (Random Forest)。用于回答“**哪个因素最重要？**”以及捕捉“**非线性关系**”（如年龄的倒U型曲线）。

---

## 📂 输出文件详解 (Output Files)

代码运行后生成的 **5张图** 和 **2个表** 是论文第三部分的骨架。请按以下顺序在论文中展示。

### 📊 图表文件 (PDF)

#### 1. `Q3_Plot1_Feature_Importance.pdf` (双模型对比)

* **图表含义**：左图是随机森林的特征重要性排行，右图是统计模型的效应大小。
* **写作要点**：
* **核心发现**：**Judge Score (评委分)** 是预测粉丝投票的最强因子（左图第一）。这验证了 **"社会认同理论" (Social Proof)** —— 观众倾向于跟随专家的判断。
* **次要因子**：Age (年龄) 和 Partner (舞伴) 的重要性紧随其后。



#### 2. `Q3_Plot2_Statistical_Effects.pdf` (效应方向)

* **图表含义**：展示了各因素对粉丝投票是“加分”还是“减分”。
* **写作要点**：
* **Age (年龄)**：通常呈现负相关（柱子在左侧），说明高龄是劣势。
* **Underdog (弱者)**：如果柱子在右侧，说明“上周排名靠后”会激发粉丝的同情票；如果在左侧或接近0，说明“马太效应”（强者恒强）主导了比赛。



#### 3. `Q3_Plot3_Halo_Effect.pdf` (光环效应)

* **图表含义**：箱线图。X轴是评委分数段，Y轴是粉丝投票份额。
* **写作要点**：
* **理论引用**：**"Halo Effect" (光环效应)**。
* **分析**：随着评委分数的上升（从Very Low到Very High），粉丝投票的中位数显著上升。这证明了评委不仅决定了技术分，还引导了舆论风向。



#### 4. `Q3_Plot4_Age_Curve.pdf` (年龄曲线)

* **图表含义**：展示了年龄与粉丝投票的非线性关系。
* **写作要点**：
* **理论引用**：**"Human Capital Theory" (人力资本理论)**。
* **分析**：你可能会看到一个 **"Golden Era" (20-30岁)** 的高峰，这是体能与流行文化影响力的巅峰。而在60岁+区域可能会有一个小反弹，这是 **"Nostalgia Premium" (怀旧溢价)**。



#### 5. `Q3_Plot5_Kingmaker_Index.pdf` (造星者指数)

* **图表含义**：展示了排除明星自身因素后，**职业舞伴 (Pro Partner)** 对投票的净贡献值。
* **写作要点**：
* **理论引用**：**"Labor Value Added" (劳动增值)**。
* **分析**：排名第一的舞伴（如 Derek Hough 等）能凭空为明星带来显著的选票加成。这量化了职业舞伴的“教学能力”和“编舞能力”的价值。



### 📋 表格文件 (CSV)

* `Q3_Table1_Statistical_Factors.csv`：包含回归系数和 **P值**。论文中用来证明结论具有“统计学显著性” (Statistically Significant, p < 0.05)。
* `Q3_Table_Pro_Partner_Value.csv`：职业舞伴的详细排名表。可作为附录。

---

## 📝 论文写作逻辑框架 (Paper Structure)

建议在第三问的报告中采用以下结构：

### 3.1 数据融合与处理 (Data Fusion)

* 强调我们没有局限于单一数据，而是融合了外部数据集（External Dataset），填补了年龄和行业的缺失值，使用了 **Imputation Techniques**（插补技术）。

### 3.2 双轨建模方法 (Dual-Track Methodology)

* 解释为什么用两个模型：
* 用 **LMM (Linear Mixed Effects)** 来处理数据的层级结构（选手嵌套在赛季中，舞伴跨越赛季）。
* 用 **Random Forest** 来捕捉复杂的非线性交互作用。
* *自夸点*：这比简单的线性回归（Linear Regression）要先进得多。



### 3.3 结果分析：谁在掌控比赛？ (Findings)

* **Finding 1: 评委的隐性权力 (The Implicit Power of Judges)**
* 引用 `Q3_Plot1` 和 `Q3_Plot3`。
* 解释：评委通过打分设定了“锚点” (Anchoring)，粉丝很难完全脱离这个锚点进行独立投票。


* **Finding 2: 年龄的生理壁垒 (The Biological Barrier)**
* 引用 `Q3_Plot4`。
* 解释：虽然节目是娱乐性质，但高强度的舞蹈对体能有硬性要求。


* **Finding 3: 隐形的造星者 (The Invisible Kingmakers)**
* 引用 `Q3_Plot5`。
* 解释：职业舞伴不仅仅是配角，他们是核心资产。



### 3.4 结论 (Conclusion)

* 回答题目："Do they impact judge scores and fan votes in the same way?"
* **答案**：**不一样**。评委更看重“行业属性”（如偏爱运动员），而粉丝更看重“光环效应”和“个人魅力”（职业舞伴的加持）。

---

## 💡 交叉学科术语库 (Buzzwords)

在写作时使用这些词汇，能提升论文的理论深度：

* **Psychology (心理学)**: Halo Effect (光环效应), Social Proof (社会认同), Anchoring Bias (锚定偏见).
* **Economics (经济学)**: Human Capital (人力资本), Value Added (附加值), Winner-Take-All (赢家通吃).
* **Sociology (社会学)**: Identity Politics (身份政治 - 粉丝支持同行业的明星), Nostalgia (怀旧情绪).
* **Statistics (统计学)**: Heterogeneity (异质性), Non-linearity (非线性), Robustness (鲁棒性).

加油！把图贴上去，配上这些理论分析，这一问稳拿分！ 💪