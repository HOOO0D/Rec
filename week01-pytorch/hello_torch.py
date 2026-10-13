import torch

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("PyTorch:", torch.__version__)
    print("Device:", device)

    # Tensor
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)

    print("Dot product:", torch.dot(a, b).item())

    # 自动求导
    x = torch.tensor(3.0, requires_grad=True, device=device)
    y = x ** 2 + 2 * x + 1

    y.backward()

    print("y:", y.item())
    print("Gradient:", x.grad.item())

if __name__ == "__main__":
    main()