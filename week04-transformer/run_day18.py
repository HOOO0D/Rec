"""Day 18 演示：Self/Cross Multi-Head Attention 与因果掩码检查。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from multihead_attention import MultiHeadAttention
from scaled_attention import make_causal_keep_mask

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--output-dir',default='experiments/day18_demo')
    args=parser.parse_args()
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA 当前不可用，请用 --device cpu')
    torch.manual_seed(18)
    mha=MultiHeadAttention(8,2,dropout_p=0.0).to(device).eval()
    x=torch.randn(2,4,8,device=device)
    causal=make_causal_keep_mask(4,device=device)
    y,weights=mha(x,keep_mask=causal,return_weights=True)
    changed=x.clone(); changed[:,3]+=100
    prefix_error=float((y[:,:3]-mha(changed,keep_mask=causal)[:,:3]).abs().max().item())
    assert prefix_error<=1e-5, 'Causal Mask 泄漏未来信息'
    memory=torch.randn(2,5,8,device=device)
    cy,cw=mha(x[:,:3],memory,memory,return_weights=True)
    assert cy.shape==(2,3,8) and cw.shape==(2,2,3,5)
    out_dir=Path(args.output_dir);out_dir.mkdir(parents=True,exist_ok=True)
    torch.save({
        'input_self':x.cpu(), 'self_mask':causal.cpu(),
        'self_output':y.detach().cpu(), 'self_weights':weights.detach().cpu(),
        'cross_output':cy.detach().cpu(), 'cross_weights':cw.detach().cpu(),
    }, out_dir/'multihead_demo.pt')
    summary={
        'day':18,'device':str(device),'training_performed':False,
        'd_model':8,'num_heads':2,'head_dim':4,
        'self_input_shape':list(x.shape),'self_output_shape':list(y.shape),
        'self_weights_shape':list(weights.shape),
        'cross_query_shape':list(x[:,:3].shape), 'cross_memory_shape':list(memory.shape),
        'cross_output_shape':list(cy.shape),'cross_weights_shape':list(cw.shape),
        'num_trainable_params':sum(p.numel() for p in mha.parameters()),
        'prefix_max_abs_error':prefix_error,
        'all_outputs_finite':bool(torch.isfinite(y).all() and torch.isfinite(cy).all()),
    }
    (out_dir/'multihead_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('已写入',out_dir.resolve())

if __name__=='__main__':main()
