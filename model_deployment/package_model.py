#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型打包脚本
将训练好的模型、预处理器、配置文件等打包，方便部署和分享
"""

import os
import shutil
import json
import pickle
import torch
from datetime import datetime
import zipfile

def package_model(model_path='lstm_model_complete.pth', 
                  output_dir='model_package',
                  include_training_log=True):
    """
    打包模型文件
    
    Parameters:
    -----------
    model_path : str
        完整模型文件路径
    output_dir : str
        输出目录
    include_training_log : bool
        是否包含训练日志
    """
    print("=" * 80)
    print("模型打包工具")
    print("=" * 80)
    print()
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        print("   请先运行训练脚本生成模型文件")
        return False
    
    # 创建输出目录
    if os.path.exists(output_dir):
        print(f"⚠️  输出目录已存在: {output_dir}")
        response = input("是否删除并重新创建? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(output_dir)
        else:
            print("取消打包")
            return False
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 创建输出目录: {output_dir}")
    
    # 加载模型检查点
    print(f"\n正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型信息
    model_info = {
        'model_name': 'LSTM_MenstrualCycle_Predictor',
        'version': '1.0.0',
        'packaged_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'LSTM模型 - 女性健康管理时间序列预测（月经时间+疼痛等级）',
        'tasks': ['menstruation_prediction', 'pain_level_prediction'],
        'input_features': [
            'emotion', 'sleep_quality', 'basal_body_temperature',
            'heart_rate', 'stress_level', 'disorder_score',
            'cumulative_disorder', 'day_in_cycle', 'phase'
        ],
        'window_size': checkpoint.get('config', {}).get('window_size', 30),
        'model_config': checkpoint.get('config', {}),
        'metrics': checkpoint.get('metrics', {}),
        'input_size': checkpoint.get('input_size', 9)
    }
    
    # 保存模型文件
    print("正在保存模型文件...")
    shutil.copy(model_path, os.path.join(output_dir, 'model.pth'))
    print("  ✓ 模型文件已保存")
    
    # 保存预处理器（如果存在）
    preprocessor_path = checkpoint.get('preprocessor_path', 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        shutil.copy(preprocessor_path, os.path.join(output_dir, 'preprocessor.pkl'))
        print("  ✓ 预处理器已保存")
    else:
        print("  ⚠️  预处理器文件不存在，请确保已运行训练脚本")
    
    # 保存模型信息
    with open(os.path.join(output_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    print("  ✓ 模型信息已保存")
    
    # 保存配置文件
    if 'config' in checkpoint:
        with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(checkpoint['config'], f, ensure_ascii=False, indent=2)
        print("  ✓ 配置文件已保存")
    
    # 保存评估指标
    if 'metrics' in checkpoint:
        with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(checkpoint['metrics'], f, ensure_ascii=False, indent=2)
        print("  ✓ 评估指标已保存")
    
    # 复制训练日志（如果存在）
    if include_training_log and os.path.exists('training.log'):
        shutil.copy('training.log', os.path.join(output_dir, 'training.log'))
        print("  ✓ 训练日志已保存")
    
    # 创建README文件
    readme_content = f"""# LSTM模型包

## 模型信息

- **模型名称**: {model_info['model_name']}
- **版本**: {model_info['version']}
- **打包日期**: {model_info['packaged_date']}
- **描述**: {model_info['description']}

## 文件说明

- `model.pth`: 训练好的模型权重和配置
- `preprocessor.pkl`: 数据预处理器（用于数据标准化和编码）
- `model_info.json`: 模型详细信息
- `config.json`: 模型配置参数
- `metrics.json`: 模型评估指标
- `predict.py`: 模型预测脚本（使用示例）
- `requirements.txt`: 依赖库列表

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 使用预测脚本

```python
from predict import MenstrualCyclePredictor

# 加载模型
predictor = MenstrualCyclePredictor('model.pth')

# 准备数据（需要30天的历史数据）
data = {{
    'emotion': [...],  # 30天的情绪得分
    'sleep_quality': [...],  # 30天的睡眠质量
    'basal_body_temperature': [...],  # 30天的基础体温
    'heart_rate': [...],  # 30天的心率
    'stress_level': [...],  # 30天的压力水平
    'disorder_score': [...],  # 30天的紊乱度
    'cumulative_disorder': [...],  # 30天的累积紊乱度
    'day_in_cycle': [...],  # 30天的周期内天数
    'phase': [...]  # 30天的周期阶段
}}

# 预测
result = predictor.predict(data)
print(f"月经概率: {{result['menstruation_probability']:.4f}}")
print(f"预测疼痛等级: {{result['pain_level']:.2f}}")
```

## 模型性能

{json.dumps(checkpoint.get('metrics', {}), ensure_ascii=False, indent=2)}

## 注意事项

1. 模型需要30天的历史数据作为输入
2. 所有特征都需要标准化处理（预测脚本会自动处理）
3. 确保输入数据的格式正确

## 技术支持

如有问题，请查看模型训练日志或联系开发者。
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("  ✓ README文件已创建")
    
    # 复制预测脚本到打包目录
    if os.path.exists('predict.py'):
        shutil.copy('predict.py', os.path.join(output_dir, 'predict.py'))
        print("  ✓ 预测脚本已复制")
    
    # 复制requirements.txt到打包目录
    if os.path.exists('requirements.txt'):
        shutil.copy('requirements.txt', os.path.join(output_dir, 'requirements.txt'))
        print("  ✓ 依赖文件已复制")
    
    # 创建压缩包
    zip_path = f"{output_dir}.zip"
    print(f"\n正在创建压缩包: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
    print(f"  ✓ 压缩包已创建: {zip_path} ({zip_size:.2f} MB)")
    
    print("\n" + "=" * 80)
    print("模型打包完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}")
    print(f"压缩包: {zip_path}")
    print(f"\n可以将压缩包发送给同事使用。")
    
    return True


if __name__ == '__main__':
    import sys
    
    # 检查命令行参数
    model_path = 'lstm_model_complete.pth'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    output_dir = 'model_package'
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    success = package_model(model_path=model_path, output_dir=output_dir)
    
    if success:
        print("\n✓ 打包成功！")
    else:
        print("\n❌ 打包失败！")
        sys.exit(1)

