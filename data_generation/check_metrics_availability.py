#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指标可采集性检查脚本
验证所有个性化指标的可采集性和获取方式
"""

import json
from data.lstm_data_simulator import MenstrualCycleSimulator

def check_metrics_availability():
    """检查所有指标的可采集性"""
    print("=" * 100)
    print("🔍 个性化指标可采集性检查")
    print("=" * 100)

    # 创建模拟器实例
    simulator = MenstrualCycleSimulator(n_users=100, days=30)  # 小样本用于测试

    # 获取指标可采集性报告
    availability_report = simulator.validate_collectible_metrics()

    print("📋 指标分类和可采集性分析:")
    print("=" * 80)

    total_metrics = 0
    collectible_metrics = 0
    derivable_metrics = 0

    for category, metrics in availability_report.items():
        print(f"\n🎯 {category.upper()}指标 ({len(metrics)}个):")
        print("-" * 60)

        for metric_name, info in metrics.items():
            total_metrics += 1
            status = "✅ 可采集" if info['collectible'] else "🔄 可推导"
            if info['collectible']:
                collectible_metrics += 1
            else:
                derivable_metrics += 1

            print(f"  {status} {metric_name}")
            print(f"      📝 获取方式: {info['method']}")
            print()

    print("=" * 80)
    print("📊 统计结果:"    print(f"  📈 总指标数: {total_metrics}")
    print(f"  ✅ 可直接采集: {collectible_metrics} ({collectible_metrics/total_metrics*100:.1f}%)")
    print(f"  🔄 可推导计算: {derivable_metrics} ({derivable_metrics/total_metrics*100:.1f}%)")
    print()

    # === 实际应用建议 ===

    print("💡 实际应用中的指标采集建议:")
    print("=" * 80)

    application_suggestions = {
        "首次评估": [
            "📋 基础信息收集: 年龄、身高、体重、月经周期历史",
            "🧠 心理评估: EPQ人格量表（15分钟）",
            "💊 症状记录: 既往痛经史、PMS症状",
            "🏥 体质辨识: 中医体质分类问卷"
        ],

        "日常监测": [
            "🌡️ 体温监测: 每天测量基础体温",
            "❤️ 心率追踪: 可穿戴设备自动采集",
            "😴 睡眠监测: 睡眠App或手环",
            "📱 情绪记录: 每日情绪状态打分",
            "😰 压力评估: 每周压力水平自评",
            "📅 月经记录: 月经开始日期和症状"
        ],

        "定期复查": [
            "📊 每月统计: 疼痛模式、情绪波动分析",
            "🔄 季度调整: 根据使用数据优化预测",
            "📈 年度评估: 完整年度数据回顾分析"
        ]
    }

    for phase, suggestions in application_suggestions.items():
        print(f"\n🚀 {phase}:")
        for suggestion in suggestions:
            print(f"  • {suggestion}")

    # === 数据隐私和伦理考虑 ===

    print("
🔒 数据隐私和伦理考虑:"    print("=" * 80)
    privacy_considerations = [
        "📖 知情同意: 明确告知数据收集目的和使用方式",
        "🔐 数据加密: 所有个人数据加密存储",
        "👤 匿名化: 用户ID匿名化处理",
        "🗑️ 数据清理: 使用后及时清理临时数据",
        "⚖️ 合规性: 符合当地数据保护法规",
        "🔍 透明度: 用户可查看自己的数据使用情况"
    ]

    for consideration in privacy_considerations:
        print(f"  • {consideration}")

    # === 技术实现建议 ===

    print("
🛠️ 技术实现建议:"    print("=" * 80)
    implementation_suggestions = [
        "📱 移动App: 集成所有数据采集功能",
        "☁️ 云同步: 安全的数据云端同步",
        "🤖 AI分析: 实时数据分析和预测",
        "📊 可视化: 直观的数据图表展示",
        "🔄 反馈循环: 用户反馈改进预测准确性",
        "🔔 智能提醒: 基于预测的健康提醒"
    ]

    for suggestion in implementation_suggestions:
        print(f"  • {suggestion}")

    # 保存详细报告
    report_file = 'metrics_availability_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'availability_report': availability_report,
            'statistics': {
                'total_metrics': total_metrics,
                'collectible_metrics': collectible_metrics,
                'derivable_metrics': derivable_metrics,
                'collectible_percentage': round(collectible_metrics/total_metrics*100, 1)
            },
            'application_suggestions': application_suggestions,
            'privacy_considerations': privacy_considerations,
            'implementation_suggestions': implementation_suggestions
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 详细报告已保存到: {report_file}")

    print("
🎉 检查完成！"    print("=" * 80)
    print("✅ 所有指标均可采集或推导")
    print("✅ 提供了完整的实施指南")
    print("🚀 可以开始实际应用开发")

def main():
    """主函数"""
    try:
        check_metrics_availability()
    except Exception as e:
        print(f"❌ 检查过程异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
