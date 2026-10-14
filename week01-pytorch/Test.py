
import torch

# ==============================
# A. Tensor (50分)
# ==============================

# A1: 创建 3x4 float32 Tensor，包含 0~11
X = torch.arange(12,dtype=torch.float32).reshape(3,4)  # TODO

assert X.shape == (3, 4)
assert X.dtype == torch.float32
assert X.ndim == 2
assert torch.equal(X.flatten(), torch.arange(12, dtype=torch.float32))

X2 = X.reshape(2,6)  # TODO: reshape 为 (2, 6)
assert X2.shape == (2, 6)
assert torch.equal(X2.flatten(), X.flatten())
print("[PASS] A1")

# A2: 广播
v = torch.tensor([1., 2., 3., 4.])
Y = X+v  # TODO: X 与 v 相加

assert Y.shape == (3, 4)
assert torch.allclose(Y[0], torch.tensor([1., 3., 5., 7.]))
print("[PASS] A2")

# A3: 矩阵乘法与逐元素乘法
W = torch.ones(4, 2)
Z = X@W  # TODO: X 与 W 做矩阵乘法
E = X*2  # TODO: X 中每个元素乘 2

assert Z.shape == (3, 2)
assert torch.allclose(Z[0], torch.tensor([6., 6.]))
assert torch.allclose(Z[2], torch.tensor([38., 38.]))
assert torch.allclose(E, X * 2)
print("[PASS] A3")

# A4: 用户-物品矩阵运算
P = torch.tensor([[1., 2.], [3., 4.]])
Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])

scores = P@Q.T # TODO: 计算所有用户对所有物品的得分

assert scores.shape == (2, 3)
assert torch.allclose(
    scores,
    torch.tensor([[1., 2., 3.], [3., 4., 7.]])
)
print("[PASS] A4")

# A5: 索引
row = X[1,:]     # TODO: X 的第二行
column = X[:,2]   # TODO: X 的第三列
first_two = X[:2,:]

assert row.tolist() == [4., 5., 6., 7.]
assert column.tolist() == [2., 6., 10.]
assert first_two.shape == (2, 4)
assert torch.equal(first_two, X[:2])
print("[PASS] A5")

# ==============================
# B. Autograd (40分)
# ==============================

# B1: 单变量求导
x = torch.tensor(2.0, requires_grad=True)
y = x**2+3*x  # TODO: f(x) = x^2 + 3x

# TODO: 对 y 调用 backward()
y.backward()
assert x.grad is not None
assert abs(x.grad.item() - 7.0) < 1e-6
print("[PASS] B1")

# B2: 多变量求导
w = torch.tensor([1., 2.], requires_grad=True)
u = torch.tensor([2., 3.])

f = sum((w*u)**2)  # TODO: f = sum((w * u)^2)

# TODO: 计算 f 对 w 的梯度
f.backward()
assert w.grad is not None
assert torch.allclose(w.grad, torch.tensor([8., 36.]))
print("[PASS] B2")

# B3: 梯度累积
t = torch.tensor(2.0, requires_grad=True)

# TODO: 计算 t^2 并执行 backward()
y1 = t**2
y1.backward()
# TODO: 再计算 3*t 并执行 backward()
# 注意：两次 backward 之间不要清零梯度
y2 = 3*t
y2.backward()
assert t.grad is not None
assert abs(t.grad.item() - 7.0) < 1e-6

# TODO: 清零 t.grad
t.grad.zero_()
assert abs(t.grad.item()) < 1e-6
print("[PASS] B3")

print("ALL A/B TESTS PASSED")
