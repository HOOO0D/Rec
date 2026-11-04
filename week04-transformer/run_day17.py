"""Day 17 示例运行：给定 Q/K/V，比较无 Mask、Padding Mask 和 Causal Mask。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from scaled_attention import (
    make_padding_keep_mask,
    make_causal_keep_mask,
    combine_keep_masks,
    scaled_dot_product_attention,
)


def main():
    parser=argparse.ArgumentParser(description="Day17 Attention 演示")
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--output-dir",default="experiments/day17_demo")
    args=parser.parse_args()
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无可用 CUDA；请改成 --device cpu")
    torch.manual_seed(17)
    # 这里直接使用人工给定的 Q/K/V；Day18 再学习四组可训练投影矩阵
    x=torch.tensor([
        [[1.,0.],[0.,1.],[1.,1.],[0.,0.]],
        [[0.,1.],[1.,0.],[.5,.2],[.1,.8]],
    ],device=device)
    tokens=torch.tensor([[11,12,13,0],[21,22,23,24]],device=device)
    causal=make_causal_keep_mask(4,device=device)
    pad_key=make_padding_keep_mask(tokens)
    pad_query=make_padding_keep_mask(tokens,for_query=True)
    combined=combine_keep_masks(causal,pad_key,pad_query)
    without_mask,wo_weights=scaled_dot_product_attention(x,x,x,return_weights=True)
    with_mask,weights=scaled_dot_product_attention(x,x,x,combined,return_weights=True)

    # 改变未来位置，验证第 0–2 个位置的输出保持一致
    altered=x.clone()
    altered[:,3,:]=torch.tensor([9.,-7.],device=device)
    altered_out=scaled_dot_product_attention(altered,altered,altered,combined)
    prefix_error=float((with_mask[:,:3]-altered_out[:,:3]).abs().max().item())
    assert prefix_error <= 1e-5, "Causal Mask 未能防止未来信息泄漏"
    assert torch.isfinite(with_mask).all(), "发现非有限输出"
    out_dir=Path(args.output_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    torch.save({
        "input_x":x.cpu(),"token_ids":tokens.cpu(),"keep_mask":combined.cpu(),
        "output_unmasked":without_mask.cpu(),"output_masked":with_mask.cpu(),
        "attention_weights_unmasked":wo_weights.cpu(),"attention_weights_masked":weights.cpu()
    },out_dir/"attention_demo.pt")
    summary={
        "day":17,"device":str(device),"training_performed":False,
        "input_shape":list(x.shape),"attention_shape":list(weights.shape),
        "output_shape":list(with_mask.shape),"mask_semantics":"True=allow, False=block",
        "prefix_max_abs_error":prefix_error,
        "all_outputs_finite":bool(torch.isfinite(with_mask).all().item())
    }
    (out_dir/"attention_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("已经保存：",out_dir.resolve())


if __name__=="__main__":
    main()
