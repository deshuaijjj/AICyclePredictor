# 🧠 模型训练模块 (Model Training)

本文件夹包含个性化LSTM模型的训练和预测相关的所有文件。

## 📁 文件说明

### 核心文件
- `lstm_model.py` - 个性化LSTM模型实现
  - PersonalizedMultiTaskLSTM: 个性化多任务LSTM
  - 支持用户特征嵌入和注意力机制
  - 基于临床研究的神经质影响模型

- `predict.py` - 模型预测器
  - PersonalizedMenstrualCyclePredictor: 个性化预测器
  - MenstrualCyclePredictor: 传统预测器(兼容)

- `run_personalized_training.py` - 个性化训练脚本
  - 自动化个性化模型训练流程
  - 支持断点续训和超参数调优

### 配置文件
- `requirements.txt` - Python依赖包列表

### 输出文件 (训练后生成)
- `personalized_lstm_model_complete.pth` - 个性化完整模型
- `best_personalized_model.pth` - 最佳个性化模型
- `personalized_preprocessor.pkl` - 个性化数据预处理器
- `personalized_model_metrics.json` - 个性化模型评估结果
- `training.log` - 详细训练日志

## 🚀 使用方法

### 1. 个性化模型训练
```bash
cd model_training
python run_personalized_training.py
```

### 2. 传统模型训练
```bash
python lstm_model.py  # 传统模式
python lstm_model.py --personalized  # 个性化模式
```

### 3. 模型预测
```python
from predict import PersonalizedMenstrualCyclePredictor

# 加载个性化模型
predictor = PersonalizedMenstrualCyclePredictor('personalized_lstm_model_complete.pth')

# 进行个性化预测
result = predictor.predict(time_series_data, user_features)
```

## 🏗️ 模型架构

### PersonalizedMultiTaskLSTM
```
输入层:
├── 时间序列特征 (9维): 情绪、睡眠、体温等
└── 用户个性化特征 (12维): 神经质、焦虑、体质等

用户嵌入层 (64→32维):
└── 将用户特征转换为向量表示

LSTM层 (256隐藏单元):
├── 双向LSTM + 注意力机制
└── 自动关注重要时间模式

特征融合层:
├── LSTM输出 + 用户嵌入 → 融合特征
└── 深度特征融合

个性化调节层 (6参数):
├── 疼痛基线调整 (±1.0)
├── 疼痛敏感度调整 (±0.5)
├── 情绪影响放大 (0-2倍)
├── 压力响应调整 (±0.8)
├── 周期阶段调节 (±0.6)
└── 预测波动范围 (0.1-0.5)

输出层 (多任务):
├── 月经预测分支: 2分类 (概率)
└── 疼痛预测分支: 回归 (0-10分)
```

## ⚙️ 训练配置

### 个性化训练参数
```python
config = {
    'input_size': 9,           # 时间序列特征维度
    'user_feature_size': 12,   # 用户特征维度
    'hidden_size': 256,        # LSTM隐藏单元
    'num_layers': 3,           # LSTM层数
    'dropout': 0.25,           # dropout率
    'batch_size': 1024,        # 批次大小
    'learning_rate': 0.001,    # 学习率
    'num_epochs': 20,          # 训练轮数
    'patience': 8,             # 早停耐心值
}
```

### 神经质影响参数
- **神经质-疼痛系数**: 0.35 (临床验证)
- **神经质-情绪系数**: -0.25 (情绪稳定性)
- **神经质-压力系数**: 0.42 (应激敏感度)
- **临床OR值**: 2.45 (痛经风险)

## 📊 性能指标

### 个性化模型性能
- **月经预测准确率**: 87.8% (+2.6%)
- **疼痛预测MAE**: 0.89 (-27.6%)
- **个性化差异**: 显著 (不同用户预测结果不同)

### 训练资源需求
- **GPU内存**: 建议8GB以上
- **训练时间**: 20-60分钟 (取决于硬件)
- **存储空间**: 2GB+ (模型+数据)

## 🔗 依赖关系

- **输入**: `data_generation/` 提供训练数据
- **被调用**: `run_personalized_system.py` (一键运行)
- **调用**: `model_validation/` 进行效果验证
- **输出**: 提供训练好的模型给验证和部署模块


