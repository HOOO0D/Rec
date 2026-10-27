"""Day 9：运行 RecBole 的 BPR + ml-100k，并保存实验追踪文件。

示例：
    python run_day9.py --config recbole_bpr.yaml --output-dir runs/day9_recbole_adam

本文件是可直接执行的启动器，不包含推荐模型的重新实现。
注意：run_recbole 完成训练后会自动执行一次测试集评价；不要将测试集作为调参依据。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
import shutil
from pathlib import Path


def json_default(obj):
    """将 NumPy 标量等对象转换为可序列化的结果。"""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (TypeError, ValueError):
            pass
    return str(obj)


def write_json(path: Path, content: object) -> None:
    """统一保存 UTF-8 中文可读 JSON。"""
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def run(config_file: Path, output_dir: Path) -> None:
    """使用框架 API 训练并保存模型、解析后配置及评价结果。"""
    if not config_file.is_file():
        raise FileNotFoundError(f"找不到 RecBole 配置：{config_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "saved"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 延迟导入：缺少或不兼容的 RecBole 不影响离线单元测试。
    import numpy as np
    import torch
    from recbole.config import Config
    from recbole.model.general_recommender.bpr import BPR
    from recbole.model.loss import BPRLoss
    from recbole.quick_start import run_recbole

    options = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "show_progress": False,
    }
    config_files = [str(config_file.resolve())]
    resolved = Config(
        model="BPR", dataset="ml-100k",
        config_file_list=config_files, config_dict=options,
    )
    # 开始训练前就保存完整解析后的配置，便于排查默认值覆盖问题。
    write_json(output_dir / "resolved_config.json", dict(resolved.final_config_dict))
    shutil.copy2(config_file, output_dir / "recbole_bpr.yaml")
    write_json(output_dir / "environment.json", {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "recbole": importlib.metadata.version("recbole"),
        "torch": torch.__version__,
        "numpy": np.__version__,
    })
    # 保存源码定位，不直接复制第三方源码；请在 IDE 中自行阅读函数。
    write_json(output_dir / "source_locations.json", {
        "bpr_file": inspect.getsourcefile(BPR),
        "bpr_calculate_loss_line": inspect.getsourcelines(BPR.calculate_loss)[1],
        "bpr_full_sort_predict_line": inspect.getsourcelines(BPR.full_sort_predict)[1],
        "bpr_loss_file": inspect.getsourcefile(BPRLoss),
        "bpr_loss_forward_line": inspect.getsourcelines(BPRLoss.forward)[1],
    })
    print("运行模型：RecBole BPR，数据集：ml-100k（框架内置）")
    print("重要：RecBole 内部重新随机划分数据，不等于 Day 7 的冻结划分。")
    result = run_recbole(
        model="BPR", dataset="ml-100k",
        config_file_list=config_files, config_dict=options,
        saved=True,
    )
    # API 本身会执行测试集评价；此处仅保存原始结果，不用于选择超参数。
    write_json(output_dir / "run_result.json", result)
    print(f"实验完成：{output_dir.resolve()}")
    print("请打开 source_locations.json 定位 calculate_loss，再完成协议审计。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 9 RecBole BPR 实验启动器")
    parser.add_argument("--config", default="recbole_bpr.yaml")
    parser.add_argument("--output-dir", default="runs/day9_recbole_adam")
    args = parser.parse_args()
    run(Path(args.config), Path(args.output_dir))


if __name__ == "__main__":
    main()
