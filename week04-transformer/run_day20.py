"""Day20 运行演示：每层独立 KV-Cache，验证增量结果与完整前向一致。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from embeddings_position import TransformerInputEmbedding
from kv_cache import CachedCausalSelfAttention, compare_full_and_cached


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default='./experiments/day20_demo')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=ap.parse_args()
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('当前 PyTorch 未检测到 CUDA')
    torch.manual_seed(20261105)
    out_dir=Path(args.output_dir);out_dir.mkdir(parents=True,exist_ok=True)
    tokens=torch.tensor([[3,4,5,6,2],[8,7,6,5,2]],dtype=torch.long,device=device)
    embed=TransformerInputEmbedding(vocab_size=32,d_model=8,max_len=32).to(device).eval()
    layers=[CachedCausalSelfAttention(8,2).to(device).eval() for _ in range(2)]
    with torch.no_grad():
        x_full=embed(tokens,start_pos=0)
        prefill=embed(tokens[:,:3],start_pos=0)
        dec1=embed(tokens[:,3:4],start_pos=3)
        dec2=embed(tokens[:,4:5],start_pos=4)
        emb_err=float((torch.cat([prefill,dec1,dec2],1)-x_full).abs().max().item())
        full,inc,err=compare_full_and_cached(layers,x_full,prefill_len=3)
    assert emb_err<=1e-5,f'位置编码切片不等价：{emb_err}'
    assert err<=1e-5,f'缓存前向不等价：{err}'
    summary={
        'day':20,'seed':20261105,'device':device,'dtype':str(x_full.dtype),
        'batch_size':2,'n_layers':2,'n_heads':2,'d_model':8,'head_dim':4,
        'prefill_len':3,'total_tokens':5,
        'cache_lengths_after_prefill_and_steps':[3,4,5],
        'attention_weights_shapes':[[2,2,3,3],[2,2,1,4],[2,2,1,5]],
        'position_max_abs_error':emb_err,'full_vs_cached_max_abs_error':err,
        'qkv_projection_tokens_without_cache':sum(range(1,6)),
        'qkv_projection_tokens_with_cache':5,
        'note':'演示是无 Padding 的双层因果注意力推理，不是实际语言模型基准性能测试',
    }
    torch.save({'tokens':tokens.cpu(),'full':full.cpu(),'cached':inc.cpu()},out_dir/'day20_outputs.pt')
    (out_dir/'day20_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
