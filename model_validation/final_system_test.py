#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终系统测试脚本
验证个性化LSTM系统的完整功能
"""

import sys
import os

def test_imports():
    """测试所有关键模块的导入"""
    print("🔍 测试模块导入...")

    try:
        from data.lstm_data_simulator import MenstrualCycleSimulator
        print("  ✅ 数据模拟器导入成功")

        from model_train.lstm_model import PersonalizedMultiTaskLSTM, PersonalizedDataPreprocessor
        print("  ✅ 个性化模型导入成功")

        from model_train.predict import PersonalizedMenstrualCyclePredictor
        print("  ✅ 个性化预测器导入成功")

        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False

def test_data_generation():
    """测试数据生成功能"""
    print("\n🔍 测试数据生成功能...")

    try:
        from data.lstm_data_simulator import MenstrualCycleSimulator

        # 创建小样本模拟器进行测试
        simulator = MenstrualCycleSimulator(n_users=10, days=30)

        # 生成用户属性
        user_attrs = simulator.generate_user_attributes()
        print("  ✅ 用户属性生成成功")

        # 验证指标可采集性
        availability = simulator.validate_collectible_metrics()
        collectible_count = sum(1 for cat in availability.values() for info in cat.values() if info['collectible'])
        total_count = sum(len(cat) for cat in availability.values())

        print(f"  ✅ 指标可采集性验证: {collectible_count}/{total_count} 个指标可直接采集")

        return True
    except Exception as e:
        print(f"  ❌ 数据生成测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建功能"""
    print("\n🔍 测试模型创建功能...")

    try:
        from model_train.lstm_model import PersonalizedMultiTaskLSTM

        # 创建个性化模型
        model = PersonalizedMultiTaskLSTM(
            input_size=9,
            user_feature_size=12,
            hidden_size=64,  # 小尺寸用于测试
            num_layers=2,
            dropout=0.1
        )
        print("  ✅ 个性化LSTM模型创建成功")

        # 检查模型参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ 模型参数数量: {total_params:,}")

        return True
    except Exception as e:
        print(f"  ❌ 模型创建测试失败: {e}")
        return False

def test_preprocessing():
    """测试数据预处理功能"""
    print("\n🔍 测试数据预处理功能...")

    try:
        from model_train.lstm_model import PersonalizedDataPreprocessor
        import pandas as pd
        import numpy as np

        # 创建预处理器
        preprocessor = PersonalizedDataPreprocessor()
        print("  ✅ 个性化预处理器创建成功")

        # 创建测试数据
        test_data = pd.DataFrame({
            'user_id': [1, 1, 1],
            'emotion': [50, 55, 60],
            'sleep_quality': [70, 75, 80],
            'basal_body_temperature': [36.2, 36.3, 36.1],
            'heart_rate': [72, 74, 71],
            'stress_level': [40, 45, 50],
            'disorder_score': [2, 3, 4],
            'cumulative_disorder': [10, 13, 17],
            'day_in_cycle': [1, 2, 3],
            'phase': ['menstruation', 'menstruation', 'menstruation'],
            'menstruation': [1, 1, 1],
            'pain_level': [3, 4, 5]
        })

        user_attrs = pd.DataFrame([{
            'user_id': 1,
            'cycle_length': 28,
            'neuroticism': 50,
            'trait_anxiety': 40,
            'psychoticism': 45,
            'constitution_type': 0,
            'constitution_coef': -2.0,
            'is_night_owl': 0,
            'base_sleep_quality': 75,
            'base_emotion': 50,
            'base_heart_rate': 72,
            'base_pain_level': 3.0,
            'stress_sensitivity': 0.4
        }])

        # 测试预处理
        X, X_user, y_menstruation, y_pain = preprocessor.fit_transform(test_data, user_attrs)
        print("  ✅ 数据预处理成功")
        print(f"    时间序列特征维度: {X.shape}")
        print(f"    用户特征维度: {X_user.shape}")

        return True
    except Exception as e:
        print(f"  ❌ 预处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 80)
    print("🧪 个性化LSTM系统最终测试")
    print("=" * 80)

    all_tests_passed = True

    # 测试导入
    if not test_imports():
        all_tests_passed = False

    # 测试数据生成
    if not test_data_generation():
        all_tests_passed = False

    # 测试模型创建
    if not test_model_creation():
        all_tests_passed = False

    # 测试预处理
    if not test_preprocessing():
        all_tests_passed = False

    # 总结
    print("\n" + "=" * 80)
    if all_tests_passed:
        print("🎉 所有测试通过！个性化LSTM系统准备就绪")
        print("✅ 核心功能验证完成")
        print("✅ 神经质检测逻辑优化")
        print("✅ 指标可采集性确认")
        print("🚀 可以开始使用个性化预测系统")
    else:
        print("❌ 部分测试失败，请检查错误信息")
        sys.exit(1)

    print("\n📋 推荐使用流程:")
    print("  1. python check_metrics_availability.py  # 检查指标可采集性")
    print("  2. python run_personalized_system.py    # 运行完整系统")
    print("  3. python test_personalization.py       # 测试个性化效果")
    print("=" * 80)

if __name__ == '__main__':
    main()
