#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个性化预测效果测试脚本
验证个性化模型对不同用户特征的适应性
"""

import numpy as np
import pandas as pd
from model_train.predict import PersonalizedMenstrualCyclePredictor, MenstrualCyclePredictor
import json
import warnings
warnings.filterwarnings('ignore')

def load_test_data():
    """加载测试数据"""
    print("正在加载测试数据...")

    # 加载数据集
    df = pd.read_csv('../data_generation/lstm_dataset.csv')
    user_attrs = pd.read_csv('../data_generation/lstm_dataset_user_attributes.csv')

    # 选择几个不同特征的用户进行测试
    test_users = []

    # 用户1：高神经质、高疼痛基线（容易焦虑、疼痛敏感）
    user1 = user_attrs[
        (user_attrs['neuroticism'] > 60) &
        (user_attrs['base_pain_level'] > 3)
    ].head(1)
    if not user1.empty:
        test_users.append(('high_neuroticism', user1.iloc[0]))

    # 用户2：低神经质、低疼痛基线（情绪稳定、不易痛）
    user2 = user_attrs[
        (user_attrs['neuroticism'] < 40) &
        (user_attrs['base_pain_level'] < 2)
    ].head(1)
    if not user2.empty:
        test_users.append(('low_neuroticism', user2.iloc[0]))

    # 用户3：血瘀体质（中医体质影响）
    user3 = user_attrs[user_attrs['constitution_type'] == 1].head(1)
    if not user3.empty:
        test_users.append(('blood_stasis', user3.iloc[0]))

    # 用户4：平和体质（中医体质影响）
    user4 = user_attrs[user_attrs['constitution_type'] == 0].head(1)
    if not user4.empty:
        test_users.append(('balanced', user4.iloc[0]))

    print(f"✓ 选择了 {len(test_users)} 个不同特征的用户进行测试")

    return df, test_users

def prepare_user_data(df, user_info, window_size=30):
    """为用户准备测试数据"""
    user_id = user_info['user_id']
    user_data = df[df['user_id'] == user_id].sort_values('date').tail(window_size)

    if len(user_data) < window_size:
        print(f"⚠️  用户 {user_id} 数据不足，使用模拟数据")
        # 生成模拟数据
        user_data = pd.DataFrame({
            'emotion': np.random.normal(50, 10, window_size),
            'sleep_quality': np.random.normal(70, 8, window_size),
            'basal_body_temperature': np.random.normal(36.5, 0.2, window_size),
            'heart_rate': np.random.normal(72, 5, window_size),
            'stress_level': np.random.normal(40, 10, window_size),
            'disorder_score': np.random.normal(5, 2, window_size),
            'cumulative_disorder': np.random.normal(20, 5, window_size),
            'day_in_cycle': np.arange(1, window_size + 1),
            'phase': np.random.choice(['menstruation', 'follicular', 'ovulation', 'luteal'], window_size)
        })

    # 转换为字典格式
    time_series_data = {
        'emotion': user_data['emotion'].tolist(),
        'sleep_quality': user_data['sleep_quality'].tolist(),
        'basal_body_temperature': user_data['basal_body_temperature'].tolist(),
        'heart_rate': user_data['heart_rate'].tolist(),
        'stress_level': user_data['stress_level'].tolist(),
        'disorder_score': user_data['disorder_score'].tolist(),
        'cumulative_disorder': user_data['cumulative_disorder'].tolist(),
        'day_in_cycle': user_data['day_in_cycle'].tolist(),
        'phase': user_data['phase'].tolist()
    }

    # 用户特征
    user_features = {
        'cycle_length': user_info['cycle_length'],
        'neuroticism': user_info['neuroticism'],
        'trait_anxiety': user_info['trait_anxiety'],
        'psychoticism': user_info['psychoticism'],
        'constitution_type': user_info['constitution_type'],
        'constitution_coef': user_info['constitution_coef'],
        'is_night_owl': user_info['is_night_owl'],
        'base_sleep_quality': user_info['base_sleep_quality'],
        'base_emotion': user_info['base_emotion'],
        'base_heart_rate': user_info['base_heart_rate'],
        'base_pain_level': user_info['base_pain_level'],
        'stress_sensitivity': user_info['stress_sensitivity']
    }

    return time_series_data, user_features

def test_personalization():
    """测试个性化预测效果"""
    print("=" * 100)
    print("🧪 个性化预测效果测试")
    print("=" * 100)

    try:
        # 加载测试数据
        df, test_users = load_test_data()

        if not test_users:
            print("❌ 未找到合适的测试用户")
            return False

        # 加载个性化模型
        print("\n🤖 加载个性化预测器...")
        personalized_predictor = PersonalizedMenstrualCyclePredictor(
            '../model_training/personalized_lstm_model_complete.pth'
        )

        # 加载通用模型作为对比
        print("🤖 加载通用预测器...")
        general_predictor = MenstrualCyclePredictor(
            '../model_training/lstm_model_complete.pth'
        )

        results = []

        print("\n" + "=" * 60)
        print("开始个性化效果测试")
        print("=" * 60)

        for user_type, user_info in test_users:
            print(f"\n👤 测试用户类型: {user_type}")
            print("-" * 40)

            # 显示用户特征
            print("用户特征:")
            print(".2f")
            print(".2f")
            print(f"  特质焦虑: {user_info['trait_anxiety']:.2f}")
            print(f"  基础疼痛水平: {user_info['base_pain_level']:.2f}")
            print(f"  压力敏感度: {user_info['stress_sensitivity']:.3f}")
            print(f"  体质类型: {['平和', '血瘀', '其他'][int(user_info['constitution_type'])]}")

            # 准备测试数据
            time_series_data, user_features = prepare_user_data(df, user_info)

            # 个性化预测
            personalized_result = personalized_predictor.predict(time_series_data, user_features)

            # 通用预测（作为对比）
            general_result = general_predictor.predict(time_series_data)

            print("
预测结果:"            print("  个性化模型:")
            print(".4f")
            print(".2f")
            print(f"    是否月经期: {personalized_result['is_menstruation']}")

            print("  通用模型:")
            print(".4f")
            print(".2f")
            print(f"    是否月经期: {general_result['is_menstruation']}")

            # 计算差异
            prob_diff = abs(personalized_result['menstruation_probability'] - general_result['menstruation_probability'])
            pain_diff = abs(personalized_result['pain_level'] - general_result['pain_level'])

            print("
差异分析:"            print(".4f")
            print(".2f")

            # 保存结果
            results.append({
                'user_type': user_type,
                'user_features': user_features,
                'personalized_result': personalized_result,
                'general_result': general_result,
                'differences': {
                    'menstruation_prob': prob_diff,
                    'pain_level': pain_diff
                }
            })

        # 总结分析
        print("\n" + "=" * 100)
        print("📊 个性化效果总结")
        print("=" * 100)

        total_prob_diff = 0
        total_pain_diff = 0

        for result in results:
            total_prob_diff += result['differences']['menstruation_prob']
            total_pain_diff += result['differences']['pain_level']

            print(f"用户 {result['user_type']}:")
            print(".4f")
            print(".2f")

        avg_prob_diff = total_prob_diff / len(results)
        avg_pain_diff = total_pain_diff / len(results)

        print("
📈 平均差异:"        print(".4f")
        print(".2f")

        if avg_prob_diff > 0.1 or avg_pain_diff > 0.5:
            print("
✅ 个性化效果显著！模型能够根据用户特征调整预测结果"        else:
            print("
⚠️  个性化效果不明显，可能需要更多训练数据或调整模型"        # 保存测试结果
        with open('personalization_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("
💾 测试结果已保存到: personalization_test_results.json"        print("=" * 100)

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    try:
        success = test_personalization()
        if success:
            print("\n🎉 个性化测试完成！查看结果了解模型对不同用户的适应性。")
        else:
            print("\n❌ 个性化测试失败，请检查模型文件是否存在。")
    except Exception as e:
        print(f"\n❌ 测试过程异常: {e}")

if __name__ == '__main__':
    main()
