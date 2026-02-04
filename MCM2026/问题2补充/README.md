

本模块针对 **Question 2** 的两个核心子问题进行了数学建模和代码实现：

1. **评委拯救规则（Judges' Save）**：分析评委从倒数两名中选择保留一对选手的规则，对比赛结果的具体影响。
2. **赛制对比（Rank vs. Percent）**：对比S1-S2使用的“排名制”与S3+使用的“百分比制”在结果上的差异及其对“技术vs人气”的偏好。

该部分代码通过 **蒙特卡洛反事实模拟（Monte Carlo Counterfactual Simulation）**，在缺失真实粉丝投票数据的情况下，重构了潜在的投票分布，并以此评估不同规则的影响。

---

```

运行结束后，将生成以下关键文件，请查收：

* 📊 **PDF 图表**：`judge_save_analysis.pdf`, `method_comparison_visuals.pdf`
* 💾 **CSV 数据**：`judge_save_impact.csv`, `method_comparison.csv`

---

## 📝 分析部分详解与解读指南

### 1. 评委拯救规则影响分析 (The Impact of Judges' Save)

**代码逻辑**：
我们无法知道真实的粉丝票数，但知道谁被淘汰了。模型生成了成千上万种“导致该选手被淘汰”的合法粉丝投票分布，然后在这些分布上应用“评委拯救”规则（即评委在倒数两名中救回评分较高的那个）。

**📄 输出文件**：`judge_save_analysis.pdf`

**图表解读（写作重点）**：

* **左图 (Scatter Plot - The Safety Net)**：
* **X轴**：选手的评委打分 Z-Score（越往右技术越强）。
* **Y轴**：翻转概率（Flip Prob），即被评委救回来的概率。
* **结论**：你可以看到一个明显的上升趋势。**这个规则充当了“技术流选手的安全网”（Safety Net）**。它专门保护那些跳得好（Z-score高）但因为人气低（导致掉入Bottom 2）的选手，防止他们过早出局。


* **右图 (Histogram)**：
* **结论**：平均约 **21%** 的周次结果会因为此规则改变。说明这个规则不是摆设，它显著修正了纯人气主导的偏差。



### 2. 排名制 vs 百分比制对比 (Rank vs Percentage Method)

**代码逻辑**：
在相同的粉丝投票和评委打分下，同时模拟两套规则：

* **Rank Method**：`Rank_Judge + Rank_Fan` (总和最大者淘汰)
* **Percent Method**：`%_Judge + %_Fan` (总和最小者淘汰)

**📄 输出文件**：`method_comparison_visuals.pdf`

**图表解读（写作重点 - 这里有一个反直觉的亮点）**：

* **左图 (Histogram - Disagreement Rate)**：
* **结论**：两种赛制在约 **24%** 的情况下会产生不同的淘汰者。赛制的改变对结果有重大影响。


* **右图 (Scatter Plot - Meritocracy Check)**：
* **X轴**：排名制淘汰者的评委排名（数值越大=跳得越差）。
* **Y轴**：百分比制淘汰者的评委排名。
* **现象**：点主要分布在对角线下方。这意味着 `Rank Method > Percent Method`。
* **深度结论（Key Insight）**：
* **Rank Method 更“精英主义”**：它倾向于淘汰评委排名差的人（Avg Rank ~7.4）。
* **Percent Method 更“民粹主义”**：它倾向于让评委排名差的人存活（Avg Rank ~6.7）。
* **解释**：在百分比制下，如果一个明星人气极高（比如获得50%粉丝票），他可以填补巨大的评委分数坑；而在排名制下，粉丝投票第一名只是Rank 1，优势被封顶了，无法无限填补技术分数的劣势。**所以，S3引入百分比制，实际上保护了高人气的“差生”。**





---

## 💡 论文写作建议 (For Paper Writing)

在 **Problem C - Q2** 的部分，建议按照以下逻辑组织文字：

1. **建立模型**：简述我们使用了基于Dirichlet分布的蒙特卡洛模拟来重构未知的粉丝数据。
2. **分析 Save 规则**：
* 引用 `judge_save_analysis.pdf`。
* 提出 **"The Safety Net Hypothesis"**：证明该规则有效防止了“劣币驱逐良币”（人气高的差选手挤走没人气的好选手）。


3. **对比赛制**：
* 引用 `method_comparison_visuals.pdf`。
* 提出 **"The Populism Shift"**：指出从 Rank 变为 Percent，实际上是赋予了粉丝投票更大的权重（uncapped leverage），使得高人气选手更难被淘汰。这解释了为什么后续赛季可能出现更多“分数不高但走得很远”的争议案例（如Jerry Rice现象的逆向验证）。

