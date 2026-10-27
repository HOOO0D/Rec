from recbole.quick_start import run_recbole

if __name__ == "__main__":
    # 读取指定的 YAML 配置
    # RecBole 会自动完成数据加载、模型训练及评价
    result = run_recbole(
        model="BPR",
        dataset="ml-100k",
        config_file_list=["./week02-bpr/recbole_baseline/configs/bpr_ml100k.yaml"]
    )

    # 显示本次实验的验证和测试结果
    print("最优验证结果：")
    print(result["best_valid_result"])

    print("最终测试结果：")
    print(result["test_result"])