#  数据生成模块 (Data Generation)

本文件夹包含所有数据生成和预处理相关的文件，用于创建个性化LSTM模型的训练数据。

##  文件说明

### 核心文件
- `lstm_data_simulator.py` - 临床验证的数据模拟器
  - 基于医学研究的神经质影响模型
  - 生成12个可采集的用户个性化指标
  - 支持临床合理性验证

- `check_metrics_availability.py` - 指标可采集性检查脚本
  - 验证所有指标的可采集性和获取方式
  - 生成详细的实施指南和隐私考虑

### 数据文件
- `lstm_dataset.csv` - 训练数据集 (720万条记录)
- `lstm_dataset_user_attributes.csv` - 用户个性化特征数据
- `dataset_summary.json` - 数据集统计信息
- `data_validation_report.json` - 临床合理性验证报告

##  使用方法

### 1. 生成训练数据
```bash
cd data_generation
python lstm_data_simulator.py
```

### 2. 检查指标可采集性
```bash
python check_metrics_availability.py
```

### 3. 查看数据验证报告
```bash
cat data_validation_report.json
```

##  指标体系

### 可直接采集指标 (6个)
1. **cycle_length** - 周期长度 (用户历史记录)
2. **neuroticism** - 神经质得分 (EPQ量表)
3. **trait_anxiety** - 特质焦虑 (STAI量表)
4. **psychoticism** - 精神质得分 (EPQ量表)
5. **constitution_type** - 体质类型 (中医问卷)
6. **is_night_owl** - 夜猫子类型 (睡眠调查)

### 可推导指标 (6个)
7. **constitution_coef** - 体质系数
8. **stress_sensitivity** - 压力敏感度
9. **base_sleep_quality** - 基础睡眠质量
10. **base_emotion** - 基础情绪水平
11. **base_heart_rate** - 基础心率
12. **base_pain_level** - 个人疼痛基线

##  临床验证

- **神经质检测**: 基于OR=2.45的meta分析
- **相关性验证**: 自动验证指标间临床相关性
- **分布合理性**: 确保数据符合医学规律

##  数据规格

- **用户数量**: 10,000个
- **时间跨度**: 720天 (24个月经周期)
- **总记录数**: 7,200,000条
- **特征维度**: 21个 (9时间序列 + 12用户特征)

##  依赖关系

- **被调用**: `run_personalized_system.py`
- **调用**: `model_training/lstm_model.py`
- **输出**: 提供训练数据给模型训练模块


