#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个性化LSTM模型训练脚本
运行个性化训练：基于用户特征的精准预测系统
"""

import os
import sys
import subprocess
from datetime import datetime

def run_personalized_training():
    """运行个性化LSTM训练"""
    print("=" * 100)
    print("🎯 个性化LSTM模型训练")
    print("基于用户个性化特征的精准预测系统")
    print("=" * 100)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 检查必要的文件
    required_files = [
        'data/lstm_dataset.csv',
        'data/lstm_dataset_user_attributes.csv',
        'model_train/lstm_model.py'
    ]

    print("🔍 检查必要文件...")
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path} (文件不存在)")

    if missing_files:
        print("
❌ 缺少必要文件:"        for file in missing_files:
            print(f"  - {file}")
        print("\n请先运行以下命令生成数据:")
        print("  cd data")
        print("  python lstm_data_simulator.py")
        return False

    print("✅ 所有必要文件检查通过")
    print()

    # 运行个性化训练
    print("🚀 开始个性化训练...")
    print("这将训练一个能够根据用户个性化特征进行精准预测的模型")
    print("训练过程可能需要较长时间，请耐心等待...")
    print()

    try:
        # 运行个性化训练
        result = subprocess.run([
            sys.executable, 'model_train/lstm_model.py', '--personalized'
        ], capture_output=True, text=True, encoding='utf-8')

        if result.returncode == 0:
            print("\n" + "=" * 100)
            print("🎉 个性化训练成功完成！")
            print("=" * 100)
            print("生成的文件:")
            print("  🧠 最佳个性化模型: model_train/best_personalized_model.pth")
            print("  🧠 完整个性化模型: model_train/personalized_lstm_model_complete.pth")
            print("  📊 个性化评估结果: model_train/personalized_model_metrics.json")
            print("  📈 训练历史图: model_train/personalized_training_history.png")
            print("  🔧 个性化预处理器: model_train/personalized_preprocessor.pkl")
            print("  📝 训练日志: model_train/training.log")
            print()
            print("📋 个性化模型特点:")
            print("  • 基于12个用户个性化指标进行预测")
            print("  • 支持注意力机制，关注重要时间特征")
            print("  • 特征融合：LSTM输出 + 用户嵌入")
            print("  • 个性化调节：根据用户特征调整预测结果")
            print()
            print("🔮 使用个性化预测:")
            print("  from model_train.predict import PersonalizedMenstrualCyclePredictor")
            print("  predictor = PersonalizedMenstrualCyclePredictor('model_train/personalized_lstm_model_complete.pth')")
            print("  result = predictor.predict(time_series_data, user_features)")
            print("=" * 100)
            return True
        else:
            print("\n❌ 个性化训练失败")
            print("错误信息:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"\n❌ 训练过程异常: {e}")
        return False

def main():
    """主函数"""
    try:
        success = run_personalized_training()
        if success:
            print("\n⭐ 个性化训练流程完成！现在您有了一个能够根据用户个性化特征进行精准预测的AI系统。")
        else:
            print("\n❌ 个性化训练失败，请检查错误信息并重试。")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断训练")
        print("您可以稍后重新运行脚本")
    except Exception as e:
        print(f"\n❌ 脚本执行异常: {e}")
        print("请检查错误信息并重试")
        sys.exit(1)

if __name__ == '__main__':
    main()
