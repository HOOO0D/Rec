# Day 19｜Encoder / Decoder 结构图与数据流作业

请使用 Markdown、Mermaid 或 ASCII 独立画图；不要直接粘贴论文原图。

## 1. Encoder Layer

【待填写】

要求标注：输入 `[B,S,d]`、Self-Attention、两个 Add & Norm、FFN、输出 `[B,S,d]`、Padding Mask 来源。

## 2. Decoder Layer

【待填写】

要求标注：Decoder 输入 `[B,T,d]`、Masked Self-Attention、Cross-Attention、FFN、三个 Add & Norm、Encoder memory 输入 `[B,S,d]`。

## 3. Cross-Attention 数据流

【待填写】

要求：明确 `Q` 来自 Decoder，`K/V` 来自 Encoder；注意力权重 `[B,h,T,S]`。请同时解释 Causal Mask 与 Source Padding Mask 分别作用在哪里。

## 4. Linear / Softmax / Shifted Right

【待填写】

要求：给出词表投影 logits `[B,T,V]`、推理时 Softmax 概率以及训练中 CrossEntropyLoss 的输入；说明右移目标序列的目的。

## 5. Post-LN 与 Pre-LN 结构对照

【待填写】

要求：明确原论文使用 Post-LN，说明 LN 放在残差和之后还是子层输入之前。

## 6. 结论

【待填写】

请用一段文字归纳：Encoder 与 Decoder 的最重要结构差异，以及哪些部分在 SASRec 中仍会用到。
