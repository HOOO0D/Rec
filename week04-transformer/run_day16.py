"""Day 16 最小前向演示，不训练 Transformer、不下载数据。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from embeddings_position import TransformerInputEmbedding, make_valid_mask, shift_right


def main():
    parser = argparse.ArgumentParser(description="Day16 输入表示演示")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/day16_demo"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 环境不可使用 CUDA，改用 --device cpu")
    torch.manual_seed(20261102)

    # 玩具词表约定：0=<PAD>；1=<BOS>；2=<EOS>；其余为普通 Token。
    src_ids = torch.tensor([[3, 4, 5, 2, 0], [7, 8, 2, 0, 0]], dtype=torch.long, device=device)
    targets = torch.tensor([[6, 9, 2, 0, 0], [5, 2, 0, 0, 0]], dtype=torch.long, device=device)
    decoder_ids = shift_right(targets, bos_id=1, padding_idx=0)
    src_encoder = TransformerInputEmbedding(vocab_size=16, d_model=8, max_len=16).to(device)
    tgt_encoder = TransformerInputEmbedding(vocab_size=16, d_model=8, max_len=16).to(device)
    # 原论文共享嵌入/输出投影权重；Day 16 只对输入路径进行演示，不审计权重共享。
    src_embedded = src_encoder(src_ids)
    tgt_embedded = tgt_encoder(decoder_ids)
    src_mask = make_valid_mask(src_ids)
    tgt_mask = make_valid_mask(decoder_ids)

    assert src_embedded.shape == (2, 5, 8)
    assert tgt_embedded.shape == (2, 5, 8)
    assert torch.isfinite(src_embedded).all() and torch.isfinite(tgt_embedded).all()
    assert (src_embedded[~src_mask] == 0).all()
    assert (tgt_embedded[~tgt_mask] == 0).all()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "task": "Day 16 Transformer 输入表示与右移输入演示",
        "device": device,
        "vocab_size": 16,
        "d_model": 8,
        "source_shape": list(src_embedded.shape),
        "decoder_shape": list(tgt_embedded.shape),
        "source_padding_count": int((~src_mask).sum().item()),
        "decoder_padding_count": int((~tgt_mask).sum().item()),
        "decoder_ids": decoder_ids.cpu().tolist(),
        "training": False,
        "description": "只测试输入表示；Day 17 才实现 Self-Attention 与 Causal Mask。",
    }
    (args.output_dir / "forward_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save({"src_input": src_embedded.detach().cpu(),
                "decoder_input": tgt_embedded.detach().cpu(),
                "src_mask": src_mask.cpu(),
                "decoder_mask": tgt_mask.cpu()}, args.output_dir / "forward_sample.pt")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[PASS] Day 16 输入路径演示成功；已写出前向结果。")


if __name__ == "__main__":
    main()
