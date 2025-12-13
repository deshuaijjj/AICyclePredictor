# 📦 模型部署模块 (Model Deployment)

本文件夹包含模型打包、部署和传统流程相关的所有文件，用于将训练好的模型打包成可部署的形式。

## 📁 文件说明

### 部署脚本
- `package_model.py` - 模型打包工具
  - 将训练好的模型打包成zip文件
  - 包含模型权重、预处理器、配置文件
  - 生成详细的使用文档和README

- `run_all.py` - 传统完整流程脚本
  - 自动化执行: 数据生成 → 模型训练 → 模型打包
  - 支持断点续传和错误恢复
  - 详细的进度显示和日志记录

### 输出文件 (打包后生成)
- `model_package.zip` - 完整的模型包
  - `model.pth` - 模型权重文件
  - `preprocessor.pkl` - 数据预处理器
  - `model_info.json` - 模型信息
  - `config.json` - 配置参数
  - `metrics.json` - 评估指标
  - `README.md` - 使用说明

## 🚀 使用方法

### 1. 模型打包
```bash
cd model_deployment
python package_model.py
```

### 2. 传统完整流程
```bash
python run_all.py
```

### 3. 解压和使用模型包
```bash
# 解压模型包
unzip model_package.zip -d deployed_model/

# 进入模型目录
cd deployed_model/

# 安装依赖
pip install -r requirements.txt

# 使用模型进行预测
python predict.py
```

## 📦 模型包内容

### 标准模型包结构
```
model_package.zip/
├── model.pth                 # 模型权重
├── preprocessor.pkl          # 数据预处理器
├── model_info.json           # 模型元信息
├── config.json              # 训练配置
├── metrics.json             # 评估指标
├── predict.py               # 预测脚本
├── requirements.txt         # 依赖列表
└── README.md                # 详细使用说明
```

### 模型信息 (model_info.json)
```json
{
  "model_name": "LSTM_MenstrualCycle_Predictor",
  "version": "1.0.0",
  "packaged_date": "2024-12-13 10:54:23",
  "description": "个性化LSTM月经周期预测模型",
  "tasks": ["menstruation_prediction", "pain_level_prediction"],
  "input_features": ["emotion", "sleep_quality", "basal_body_temperature", ...],
  "window_size": 30,
  "model_config": {...},
  "metrics": {...},
  "input_size": 9
}
```

## 🔄 传统流程说明

### 完整执行流程
1. **数据生成**: 调用`data_generation/lstm_data_simulator.py`
2. **模型训练**: 调用`model_training/lstm_model.py`
3. **模型打包**: 执行`package_model.py`
4. **功能测试**: 简单验证预测功能

### 流程控制
- **自动检查**: 检查每一步的输出文件是否存在
- **错误恢复**: 某一步失败后可重新执行
- **进度显示**: 实时显示执行进度和预计时间
- **日志记录**: 详细记录每一步的执行情况

### 使用示例
```bash
# 运行完整传统流程
cd model_deployment
python run_all.py

# 输出示例:
# ================================
# 🤖 女性健康管理智能预测系统 - 完整流程
# ================================
#
# 📊 步骤1: 数据生成
# ✅ 数据生成成功 (耗时: 45.2秒)
#
# 🧠 步骤2: 模型训练
# ✅ 模型训练成功 (耗时: 1845.7秒)
#
# 📦 步骤3: 模型打包
# ✅ 模型打包成功 (耗时: 12.3秒)
#
# 🎉 完整流程执行完成！
# 总耗时: 1903.2秒 (31.7分钟)
```

## 🔧 部署配置

### 系统要求
- **Python**: 3.8+
- **内存**: 16GB+ (用于大数据集处理)
- **存储**: 100GB+ (包含数据集和模型)
- **GPU**: 可选 (用于加速训练)

### 环境配置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## 📋 部署清单

### 前置检查
- [ ] Python 3.8+ 已安装
- [ ] 足够的磁盘空间 (100GB+)
- [ ] 稳定的网络连接 (下载依赖包)
- [ ] GPU驱动正确安装 (如使用GPU)

### 部署步骤
1. [ ] 下载项目代码
2. [ ] 创建虚拟环境
3. [ ] 安装依赖包
4. [ ] 执行完整流程脚本
5. [ ] 验证模型包完整性
6. [ ] 测试预测功能

### 验证方法
```bash
# 验证模型包
cd deployed_model/
python -c "
from predict import MenstrualCyclePredictor
predictor = MenstrualCyclePredictor('model.pth')
print('✅ 模型加载成功')
"

# 运行基本预测测试
python predict.py --test
```

## 🔗 依赖关系

- **调用**: `data_generation/` (数据生成), `model_training/` (模型训练)
- **被调用**: `run_personalized_system.py` (传统流程)
- **输出**: 生成可部署的模型包供最终使用

## 📞 支持与维护

### 常见问题
1. **内存不足**: 减少batch_size或使用更小的模型
2. **训练时间过长**: 使用GPU加速或减少训练轮数
3. **模型包过大**: 使用模型压缩技术减小包体积

### 版本兼容性
- **PyTorch**: 1.10+ (支持新版本的序列化格式)
- **scikit-learn**: 1.0+ (预处理器兼容性)
- **pandas**: 1.3+ (数据处理功能)

### 更新维护
- 定期更新依赖包版本
- 监控模型性能变化
- 根据用户反馈调整模型参数


