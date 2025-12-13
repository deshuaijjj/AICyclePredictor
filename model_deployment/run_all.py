#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键运行脚本
自动执行完整项目流程：数据生成 → 模型训练 → 模型打包
"""

import os
import sys
import time
import subprocess
from datetime import datetime
import shutil


def run_command(cmd, cwd=None, description=""):
    """执行命令并显示结果"""
    print(f"\n{'='*80}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80)

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd if isinstance(cmd, list) else cmd.split(),
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {description} 成功完成 (耗时: {elapsed:.1f}秒)")
            if result.stdout:
                # 只显示最后几行输出，避免过多信息
                lines = result.stdout.strip().split('\n')
                if len(lines) > 20:
                    print("输出(最后20行):")
                    print('\n'.join(lines[-20:]))
                else:
                    print("输出:")
                    print(result.stdout)
        else:
            print(f"❌ {description} 失败 (耗时: {elapsed:.1f}秒)")
            print("错误输出:")
            print(result.stderr)
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} 异常 (耗时: {elapsed:.1f}秒)")
        print(f"异常信息: {e}")
        return False

    return True


def check_dependencies():
    """检查依赖是否安装"""
    print("🔍 检查Python依赖...")

    required_packages = [
        'numpy', 'pandas', 'scipy', 'sklearn', 'torch', 'matplotlib', 'tqdm'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} (未安装)")

    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r model_train/requirements.txt")
        return False

    print("✅ 所有依赖检查通过")
    return True


def check_files():
    """检查必要文件是否存在"""
    print("🔍 检查项目文件...")

    required_files = [
        'data/lstm_data_simulator.py',
        'model_train/lstm_model.py',
        'model_train/package_model.py',
        'model_train/predict.py',
        'model_train/requirements.txt'
    ]

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (文件不存在)")
            return False

    print("✅ 项目文件检查通过")
    return True


def main():
    """主函数"""
    print("=" * 100)
    print("🤖 女性健康管理智能预测系统 - 一键运行脚本")
    print("=" * 100)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    total_start_time = time.time()

    # 检查环境
    if not check_dependencies():
        print("\n❌ 环境检查失败，请先安装依赖")
        sys.exit(1)

    if not check_files():
        print("\n❌ 文件检查失败，请检查项目完整性")
        sys.exit(1)

    # 获取用户确认
    print("\n🚀 即将执行完整流程:")
    print("  1. 📊 数据生成 (data/lstm_data_simulator.py)")
    print("  2. 🧠 模型训练 (model_train/lstm_model.py)")
    print("  3. 📦 模型打包 (model_train/package_model.py)")
    print("  4. 🧪 模型测试 (model_train/predict.py)")
    print("\n⚠️  注意: 模型训练可能需要数小时到数天时间")

    try:
        response = input("\n是否继续? (y/n): ").lower().strip()
        if response not in ['y', 'yes', '是']:
            print("已取消执行")
            return
    except KeyboardInterrupt:
        print("\n已取消执行")
        return

    # 步骤1: 数据生成
    print("\n" + "="*100)
    print("📊 步骤1: 数据生成")
    print("="*100)

    if not run_command(
        [sys.executable, 'lstm_data_simulator.py'],
        cwd='../data_generation',
        description="数据生成"
    ):
        print("❌ 数据生成失败")
        sys.exit(1)

    # 检查数据是否生成成功
    if not os.path.exists('../data_generation/lstm_dataset.csv'):
        print("❌ 数据文件未生成")
        sys.exit(1)

    # 步骤2: 模型训练
    print("\n" + "="*100)
    print("🧠 步骤2: 模型训练")
    print("="*100)
    print("⚠️  注意: 训练过程可能需要很长时间，请耐心等待...")
    print("   您可以在训练过程中查看 training.log 文件了解进度")

    if not run_command(
        [sys.executable, 'lstm_model.py'],
        cwd='../model_training',
        description="模型训练"
    ):
        print("❌ 模型训练失败")
        sys.exit(1)

    # 检查模型是否训练成功
    if not os.path.exists('../model_training/lstm_model_complete.pth'):
        print("❌ 模型文件未生成")
        sys.exit(1)

    # 步骤3: 模型打包
    print("\n" + "="*100)
    print("📦 步骤3: 模型打包")
    print("="*100)

    if not run_command(
        [sys.executable, 'package_model.py'],
        cwd='.',
        description="模型打包"
    ):
        print("❌ 模型打包失败")
        sys.exit(1)

    # 检查打包是否成功
    if not os.path.exists('model_package.zip'):
        print("❌ 模型包未生成")
        sys.exit(1)

    # 步骤4: 模型测试（可选）
    print("\n" + "="*100)
    print("🧪 步骤4: 模型测试")
    print("="*100)

    # 复制模型包到model_train目录供测试使用
    if os.path.exists('model_train/model_package.zip'):
        try:
            # 简单测试预测功能
            test_result = run_command(
                [sys.executable, 'predict.py', '--test'],
                cwd='../model_training',
                description="模型功能测试"
            )
            if not test_result:
                print("⚠️  模型测试失败，但不影响主流程")
        except:
            print("⚠️  模型测试跳过")

    # 计算总耗时
    total_elapsed = time.time() - total_start_time
    total_hours = int(total_elapsed // 3600)
    total_minutes = int((total_elapsed % 3600) // 60)
    total_seconds = int(total_elapsed % 60)

    # 输出总结
    print("\n" + "="*100)
    print("🎉 完整流程执行完成！")
    print("="*100)
    print(f"总耗时: {total_hours}小时 {total_minutes}分钟 {total_seconds}秒")
    print("\n📁 生成的文件:")
    print("  📊 数据文件: ../data_generation/lstm_dataset.csv")
    print("  📊 数据统计: ../data_generation/dataset_summary.json")
    print("  🧠 最佳模型: ../model_training/best_model.pth")
    print("  🧠 完整模型: ../model_training/lstm_model_complete.pth")
    print("  📦 模型包: model_package.zip")
    print("  📝 训练日志: ../model_training/training.log")
    print("  📊 评估结果: ../model_training/model_metrics.json")

    print("\n🚀 下一步:")
    print("  1. 查看训练日志: cat ../model_training/training.log")
    print("  2. 查看模型性能: cat ../model_training/model_metrics.json")
    print("  3. 使用模型预测: python ../model_training/predict.py")

    print("\n⭐ 项目执行成功！感谢使用女性健康管理智能预测系统。")
    print("="*100)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
        print("您可以稍后重新运行脚本")
    except Exception as e:
        print(f"\n❌ 脚本执行异常: {e}")
        print("请检查错误信息并重试")
        sys.exit(1)

