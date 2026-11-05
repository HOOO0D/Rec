"""Day 19 演示：单层 Encoder、Decoder 与词表投影（随机参数、无训练）。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer, TokenOutputProjection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="experiments/day19_demo")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 不支持 CUDA，请改用 --device cpu")
    torch.manual_seed(19)
    encoder = TransformerEncoderLayer(8, 2, 16, dropout_p=0.0).to(device).eval()
    decoder = TransformerDecoderLayer(8, 2, 16, dropout_p=0.0).to(device).eval()
    projection = TokenOutputProjection(8, 13).to(device).eval()
    src = torch.randn(2, 5, 8, device=device)
    tgt = torch.randn(2, 4, 8, device=device)
    src_mask = torch.ones(2, 1, 5, dtype=torch.bool, device=device)
    src_mask[1, :, -1] = False
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool, device=device))
    memory = encoder(src, src_keep_mask=src_mask)
    hidden = decoder(tgt, memory, tgt_keep_mask=causal, memory_keep_mask=src_mask)
    logits = projection(hidden)
    targets = torch.tensor([[1,2,3,0],[4,5,6,0]], device=device)
    loss = F.cross_entropy(logits.reshape(-1, 13), targets.reshape(-1), ignore_index=0)
    changed = tgt.clone(); changed[:, -1] += 100
    hidden_changed = decoder(changed, memory, tgt_keep_mask=causal, memory_keep_mask=src_mask)
    prefix_error = (projection(hidden_changed)[:, :-1] - logits[:, :-1]).abs().max().item()
    assert prefix_error <= 1e-5 and torch.isfinite(loss), "因果测试或损失有限性测试失败"
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_input": src.cpu(), "decoder_input": tgt.cpu(),
        "memory": memory.detach().cpu(), "hidden": hidden.detach().cpu(),
        "logits": logits.detach().cpu(), "target": targets.cpu(),
        "causal_keep_mask": causal.cpu(), "source_keep_mask": src_mask.cpu(),
    }, folder / "day19_forward.pt")
    result = {
        "day": 19, "device": device, "training_performed": False,
        "source_shape": list(src.shape), "target_shape": list(tgt.shape),
        "encoder_output_shape": list(memory.shape),
        "decoder_output_shape": list(hidden.shape),
        "logits_shape": list(logits.shape),
        "cross_entropy": float(loss.item()),
        "prefix_max_abs_error": float(prefix_error),
        "encoder_trainable_params": sum(p.numel() for p in encoder.parameters()),
        "decoder_trainable_params": sum(p.numel() for p in decoder.parameters()),
        "projection_trainable_params": sum(p.numel() for p in projection.parameters()),
    }
    (folder / "day19_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("已保存至", folder.resolve())


if __name__ == "__main__":
    main()
