# Day 19｜Transformer Block 学习笔记

> 请用自己的话填写各项。对于不确定的结论，应标记来源与实验条件。

## 1. 论文阅读
- 阅读：Attention Is All You Need，Section 3.1、3.3、3.4 与 Figure 1。
- Post-LN 的原论文定义：【待填写】
- 与 Harvard Annotated Transformer 中 Pre-LN 实现的差异：【待填写】

## 2. 残差与 LayerNorm
- Residual Connection 有什么价值？请结合梯度链式法则说明：【待填写】
- `nn.LayerNorm(d_model)` 在 `[B,L,d_model]` 上沿哪个维度计算均值方差？【待填写】
- `gamma` 和 `beta` 的含义、参数形状：【待填写】
- 手算 x=[1,2,3,4]，F(x)=[1,0,-1,0]、eps=0 的 Post-LN：【待填写】
- 为什么不能仅凭“均值为零、方差为一”来验证 Post-LN 的完整实现？【待填写】

## 3. Position-wise FFN
- 公式与输入/隐藏/输出形状：【待填写】
- 用代码验证：修改一个 Token 不改变其他 Token 的 FFN 输出；记录数值：【待填写】
- 与 Self-Attention 的 Token 交互方式有什么根本区别？【待填写】

## 4. Encoder / Decoder
- Encoder 两个子层的次序、Mask 来源：【待填写】
- Decoder 三个子层的次序、Q/K/V 来源与 Mask：【待填写】
- `Shifted Right` 与因果 Mask 为什么需要同时存在？【待填写】

## 5. 输出层与损失
- `[B,T,d]` → `[B,T,V]` 的线性投影方式：【待填写】
- 为什么使用 CrossEntropyLoss 时不应该先 Softmax？【待填写】
- 论文提到输出权重与 Embedding 权重共享：如何满足形状约束？【待填写】

## 6. 运行记录
- 环境、PyTorch 版本、设备：【待填写】
- 自动验收 A–F 得分：【待填写】
- 演示的 source / target / logits 形状：【待填写】
- Prefix Leakage 误差：【待填写】
- 遇到的错误及解决方式：【待填写】

## 7. 与 Day 20/21 和 SASRec 的衔接
- Day 20 为什么需要缓存 K/V，而不是缓存全部历史 Q？【待填写】
- Day 21 还缺哪些完整模型组件？【待填写】
- SASRec 会复用哪些部分，哪些结构不需要？【待填写】
