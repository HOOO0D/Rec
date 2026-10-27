"""Day 9：RecBole 与独立 BPR-MF 的实验协议审计。

请独立完成三个 TODO 函数，并使用 Test_Day9.py 验收。
命令行部分负责加载两个实验目录，不要求修改。
比较报告只说明协议是否相同，不能根据不同协议的指标判定模型优劣。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 这些字段是本次实验必须审计的关键维度。
AUDIT_FIELDS = (
    "optimizer", "embedding_size", "seed", "negative_sampling",
    "split_strategy", "candidate_mode", "regularization", "id_mapping",
)


def summarize_day7(config: dict) -> dict:
    """规范化 Day 7 的 config.json。

    返回的字典需要包含 AUDIT_FIELDS 中的全部字段。

    字段约定：
    optimizer       ：优化器，写成小写（例如 sgd）
    embedding_size  ：整数维度，从 embedding_dim 读取
    seed            ：整数种子
    negative_sampling：'uniform_ds_bootstrap'
    split_strategy  ：'frozen_day7_user_split'
    candidate_mode  ：'full'，指 Day 8 全物品排序
    regularization  ：'sampled_embedding_l2'，指 Day 7 的 Batch Embedding 正则
    id_mapping      ：'zero_based_no_padding'

    最后三个固定协议来自此前 Day 7/8 的代码约定，不能仅凭原始数据集名称推断。
    """
    # TODO 1：将原始 Day 7 配置转换为标准字典；不要修改输入。
    return {
        "optimizer": config["optimizer"].lower(),
        "embedding_size": int(config["embedding_dim"]),
        "seed": int(config["seed"]),
        "negative_sampling": "uniform_ds_bootstrap",
        "split_strategy": "frozen_day7_user_split",
        "candidate_mode": "full",
        "regularization": "sampled_embedding_l2",
        "id_mapping": "zero_based_no_padding",
    }
    raise NotImplementedError("请完成 summarize_day7")


def summarize_recbole(config: dict) -> dict:
    """规范化 RecBole 的已解析配置（也支持课程 YAML 读取出的字典）。

    返回的字典必须包含 AUDIT_FIELDS 中的全部字段。

    negative_sampling：当 distribution=uniform、sample_num=1 时，记作
      'uniform_negative_per_observed_positive'，否则记作 'other'。
    split_strategy：eval_args.split 含 RS、group_by=user、order=RO 时，记作
      'recbole_user_random_ratio'，否则记作 'other'。
    candidate_mode：取 eval_args.mode，统一为小写。
    regularization：weight_decay 为 0 时记作 'none'，否则记作 'optimizer_weight_decay'。
    id_mapping：'recbole_token_mapping_with_padding'。

    请注意：BPR.calculate_loss() 并没有手写 Embedding 正则项；这一规范化仅
    用于当前指定的 RecBole BPR 基线，不应无条件推广到所有 RecBole 模型。
    """
    # TODO 2：读取嵌套的 eval_args、train_neg_sample_args，并转成统一字段。
    neg_args = config["train_neg_sample_args"]
    eval_args = config["eval_args"]

    # 是否为每条已观测正反馈均匀采样一个负样本
    uniform_sampling = (
        neg_args.get("distribution") == "uniform"
        and neg_args.get("sample_num") == 1
    )

    # 是否为按用户分组的随机比例划分
    user_random_split = (
        "RS" in eval_args.get("split", {})
        and eval_args.get("group_by") == "user"
        and eval_args.get("order") == "RO"
    )

    # YAML 中通常是字符串；RecBole 解析后可能是字典
    mode = eval_args["mode"]
    if isinstance(mode, dict):
        modes = {value.lower() for value in mode.values()}
        if len(modes) != 1:
            raise ValueError("验证集和测试集的评价模式不能合并为同一个值")
        mode = modes.pop()

    return {
        "optimizer": config["learner"].lower(),
        "embedding_size": int(config["embedding_size"]),
        "seed": int(config["seed"]),
        "negative_sampling": (
            "uniform_negative_per_observed_positive"
            if uniform_sampling else "other"
        ),
        "split_strategy": (
            "recbole_user_random_ratio"
            if user_random_split else "other"
        ),
        "candidate_mode": mode.lower(),
        "regularization": (
            "none"
            if config.get("weight_decay", 0) == 0
            else "optimizer_weight_decay"
        ),
        "id_mapping": "recbole_token_mapping_with_padding",
    }
    raise NotImplementedError("请完成 summarize_recbole")


def compare_protocols(day7: dict, recbole: dict) -> list[dict]:
    """按 AUDIT_FIELDS 顺序比较两个规范化字典。

    只输出不一致的字段，每条记录使用：
      {'field': 字段名, 'day7': Day7 值, 'recbole': RecBole 值}
    对相同字段不输出记录；不允许修改两个输入字典。
    """
    # TODO 3：逐字段比较，输出协议差异。
    differences = []
    for field in AUDIT_FIELDS:
        if day7[field] != recbole[field]:
            differences.append({
                "field": field,
                "day7": day7[field],
                "recbole": recbole[field],
            })
    return differences
    raise NotImplementedError("请完成 compare_protocols")


def main() -> None:
    """加载既有实验配置并写出协议审计报告。"""
    parser = argparse.ArgumentParser(description="Day 9 实验协议审计")
    parser.add_argument("--day7-run", required=True, help="Day 7 的 SGD 实验目录")
    parser.add_argument("--recbole-run", required=True, help="Day 9 的 RecBole 实验目录")
    args = parser.parse_args()
    day7_path = Path(args.day7_run)
    recbole_path = Path(args.recbole_run)
    with (day7_path / "config.json").open(encoding="utf-8") as f:
        day7_raw = json.load(f)
    with (recbole_path / "resolved_config.json").open(encoding="utf-8") as f:
        recbole_raw = json.load(f)

    a = summarize_day7(day7_raw)
    b = summarize_recbole(recbole_raw)
    differences = compare_protocols(a, b)
    report = {
        "day7_run": str(day7_path),
        "recbole_run": str(recbole_path),
        "day7_protocol": a,
        "recbole_protocol": b,
        "differences": differences,
        "directly_comparable": len(differences) == 0,
    }
    output = recbole_path / "protocol_comparison.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"协议审计已写入：{output}")
    print(f"发现 {len(differences)} 项不一致；协议不同不得直接比较指标高低。")
    for entry in differences:
        print(f" - {entry['field']}: Day 7={entry['day7']}；RecBole={entry['recbole']}")


if __name__ == "__main__":
    main()
