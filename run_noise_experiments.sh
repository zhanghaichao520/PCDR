#!/bin/bash

# 设置实验参数
DATASET="ml-1m"  # 数据集名称
MODELS="LightGCN MACR_LGN DICE_LGN IPS_LGN DCCL_LGN PCDR_LGN"  # 要测试的模型列表
NOISE_RATIOS="0.0 0.02 0.04 0.06 0.08 0.10 0.12 0.14 "  # 噪声比例列表
OUTPUT_DIR="~/PCDR/noise_experiments"  # 输出目录

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行实验
for model in $MODELS; do
    for noise in $NOISE_RATIOS; do
        echo "运行模型: $model, 噪声比例: $noise"

        # 创建输出文件名
        output_file="$OUTPUT_DIR/${model}_noise${noise}.log"

        # 运行实验并将输出重定向到文件
        python run.py --model=$model --dataset=$DATASET --noise_ratio=$noise > "$output_file" 2>&1

        # 从日志中提取测试结果并保存到摘要文件
        test_result=$(grep "test result" "$output_file")

        # 显示测试结果
        echo "测试结果: $test_result"
        echo "测试结果: $test_result" >> "$OUTPUT_DIR/all_results.txt"

        echo "完成: $model, 噪声比例: $noise"
        echo "----------------------------------------"

        echo "完成: $model, 噪声比例: $noise" >> "$OUTPUT_DIR/all_results.txt"
        echo "----------------------------------------" >> "$OUTPUT_DIR/all_results.txt"

    done
done

echo "所有实验完成！结果摘要保存在 $OUTPUT_DIR/all_results.txt"