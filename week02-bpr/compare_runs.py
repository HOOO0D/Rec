"""比较两次第 7 天实验，检查实验公平性，不评价算法质量。

python compare_runs.py runs/day7_sgd runs/day7_adam
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def load_run(path: Path):
    config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
    history = [json.loads(line) for line in (path / 'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    return config, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sgd_dir', type=Path)
    p.add_argument('adam_dir', type=Path)
    args = p.parse_args()
    s, sh = load_run(args.sgd_dir)
    a, ah = load_run(args.adam_dir)
    checks = {
        'SGD vs Adam optimizer': (s['optimizer'], a['optimizer']) == ('sgd', 'adam'),
        'same seed': s['seed'] == a['seed'],
        'same initial embedding weights': s['initial_weight_sha256'] == a['initial_weight_sha256'],
        'same triple-sampling sequence': [h['sample_sha256'] for h in sh] == [h['sample_sha256'] for h in ah],
        'same sampler': s['sampler'] == a['sampler'] == 'bootstrap_uniform_with_replacement',
        'same dim/batch/steps/epochs/reg': all(s[k] == a[k] for k in ('embedding_dim','batch_size','steps_per_epoch','epochs','l2_weight')),
        'same train/valid/test splits': all((args.sgd_dir / f).read_bytes() == (args.adam_dir / f).read_bytes() for f in ('train_pairs.tsv', 'valid_pairs.tsv', 'test_pairs.tsv')),
    }
    for label, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    print('\nComparison (training metrics only; not Recall@K/NDCG@K):')
    for label, hist in [('SGD', sh), ('Adam', ah)]:
        if not hist:
            raise ValueError(f'{label} run has no history')
        print(f"{label}: lr={s['lr'] if label == 'SGD' else a['lr']}, "
              f"first BPR={hist[0]['bpr_loss']:.5f}, last BPR={hist[-1]['bpr_loss']:.5f}")
    if not all(checks.values()):
        raise SystemExit('Comparison is not controlled: fix failed protocol checks')
    print('Fairness checks passed. Use Day 8 validation Top-K metrics for model selection; do not declare a winner from training loss alone.')


if __name__ == '__main__':
    main()
