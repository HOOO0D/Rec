"""Day 15：BPR-MF / LightGCN 公平对照与消融分析的独立 TODO。

请独立完成以下函数，单元测试中会提供合成的配置、验证与测试元数据。

约定：五组训练分别为 K=0,1,2,3,4；K=0 视为同实现的 MF 基线。
      先用验证 NDCG@K 选择层数，最后才对 K=0 和选中的最优层数测试。
"""
from __future__ import annotations

from collections.abc import Mapping


# 除传播层数外，五组实验必须一致的配置字段。
# 注意，这些键都真实存在于 Day 14 的 experiment_config.json 中。
FAIR_FIELDS = (
    "embedding_dim", "batch_size", "epochs", "learning_rate", "reg_lambda",
    "topk", "seed", "device", "optimizer", "weight_decay", "negative_sampling",
    "l2_convention", "eval", "train_pairs_sha256", "day7_run_dir", "day11_graph_dir",
)
LAYERS = (0, 1, 2, 3, 4)


def audit_common_protocol(configs: Mapping[int, dict]) -> dict:
    """验证所有 K=0..4 实验均存在，且仅允许 n_layers 不同。

    返回这五组实验的共享协议字典（包含 FAIR_FIELDS 中的全部字段）。
    任意配置缺失、参数不一致或层数不对应，应抛 ValueError。
    """
    if set(configs) != set(LAYERS):
        raise ValueError(f"实验组别必须恰好为 {LAYERS}")
    for layer in LAYERS:
        config = configs[layer]
        if not isinstance(config, Mapping) or config.get("n_layers") != layer:
            raise ValueError(f"K={layer} 的配置必须包含 n_layers={layer}")
    shared = {}
    for field in FAIR_FIELDS:
        for layer in LAYERS:
            config = configs[layer]
            if field not in config:
                raise ValueError(f"K={layer} 的配置缺少字段 {field}")
            if layer == 0:
                shared[field] = config[field]
            elif config[field] != shared[field]:
                raise ValueError(f"K={layer} 的字段 {field} 与 K=0 不一致")
    return shared


def extract_valid_result(layer: int, config: dict, run_summary: dict) -> dict:
    """从 Day 14 保存结果抽取统一比较记录。

    返回键：n_layers、best_epoch、recall、hit、ndcg、n_eval_users、train_pairs_sha256。
    指标键取决于 config["topk"]，如 recall@10、hit@10、ndcg@10。
    检查验证数值在 [0,1]、参与评价用户数 >0、best_epoch 合法，
    并核对 Day 14 run_summary 的训练图 SHA256 与 config 完全相同。
    """
    if not isinstance(config, Mapping) or not isinstance(run_summary, Mapping):
        raise ValueError("配置和运行摘要必须是映射")
    if (type(layer) is not int or layer not in LAYERS
            or type(config.get("n_layers")) is not int
            or config["n_layers"] != layer):
        raise ValueError("传播层数与配置不匹配")

    topk = config.get("topk")
    epochs = config.get("epochs")
    best_epoch = run_summary.get("best_epoch")
    if type(topk) is not int or topk <= 0:
        raise ValueError("topk 必须是正整数")
    if (type(epochs) is not int or epochs <= 0
            or type(best_epoch) is not int or not 1 <= best_epoch <= epochs):
        raise ValueError("best_epoch 必须是训练轮数范围内的正整数")

    digest = config.get("train_pairs_sha256")
    if (not isinstance(digest, str) or not digest
            or run_summary.get("train_pairs_sha256") != digest):
        raise ValueError("训练图 SHA256 缺失或与配置不一致")
    metrics = run_summary.get("best_valid_result")
    if not isinstance(metrics, Mapping):
        raise ValueError("缺少最佳验证结果")
    n_eval_users = metrics.get("n_eval_users")
    if type(n_eval_users) is not int or n_eval_users <= 0:
        raise ValueError("n_eval_users 必须是正整数")

    record = {
        "n_layers": layer,
        "best_epoch": best_epoch,
        "n_eval_users": n_eval_users,
        "train_pairs_sha256": digest,
    }
    for name in ("recall", "hit", "ndcg"):
        key = f"{name}@{topk}"
        value = metrics.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 <= value <= 1):
            raise ValueError(f"{key} 必须是 [0,1] 内的有限数值")
        record[name] = value
    return record


def select_best_layer(rows: list[dict]) -> int:
    """根据验证 NDCG 选择 K；并列时优先传播层数较少的模型。

    必须包含 K=0..4 恰好五组；不能使用任何 test 字段。
    """
    if (len(rows) != len(LAYERS)
            or any(type(row.get("n_layers")) is not int for row in rows)
            or {row["n_layers"] for row in rows} != set(LAYERS)):
        raise ValueError("必须包含 K=0..4 恰好五组")
    for row in rows:
        ndcg = row.get("ndcg")
        if (isinstance(ndcg, bool) or not isinstance(ndcg, (int, float))
                or not 0 <= ndcg <= 1):
            raise ValueError("ndcg 必须是 [0,1] 内的有限数值")
    return max(rows, key=lambda row: (row["ndcg"], -row["n_layers"]))["n_layers"]


def make_ablation_rows(rows: list[dict]) -> list[dict]:
    """按照 K 排序，新增 ndcg_gain_pct_vs_k0、recall_gain_pct_vs_k0。

    增益以 K=0 为基线。基线为 0 时不可以除以 0，约定返回 None。
    负增益应该保留负号，不可以把指标下降写成性能提升。
    """
    baseline = next(row for row in rows if row["n_layers"] == 0)
    result = []
    for row in sorted(rows, key=lambda row: row["n_layers"]):
        record = dict(row)
        for metric in ("ndcg", "recall"):
            base_value = baseline[metric]
            record[f"{metric}_gain_pct_vs_k0"] = (
                ((row[metric] - base_value) / base_value) * 100
                if base_value != 0 else None
            )
        result.append(record)
    return result


def audit_final_history(
    train: list[tuple[int, int]],
    valid: list[tuple[int, int]],
    test: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """最终测试可屏蔽的历史交互为 Train ∪ Valid，标签仅来自 Test。

    三份划分中的 (u,i) 必须两两不相交且各自无重复。不得把 test 放入历史；
    不得修改训练图或把 valid 放入图中。返回待屏蔽交互列表。
    """
    train_set, valid_set, test_set = set(train), set(valid), set(test)
    for name, pairs, unique_pairs in (
        ("train", train, train_set),
        ("valid", valid, valid_set),
        ("test", test, test_set),
    ):
        if len(pairs) != len(unique_pairs):
            raise ValueError(f"{name} 中存在重复交互")
    if train_set & valid_set or train_set & test_set or valid_set & test_set:
        raise ValueError("train、valid、test 必须两两不相交")
    return train + valid
