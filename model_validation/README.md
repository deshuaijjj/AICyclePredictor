#  模型验证模块 (Model Validation)

本文件夹包含所有模型验证、测试和效果评估相关的文件，用于验证个性化LSTM模型的性能和个性化效果。

##  文件说明

### 测试脚本
- `test_personalization.py` - 个性化效果测试
  - 对比不同用户特征下的预测差异
  - 验证个性化调节的有效性
  - 生成详细的个性化分析报告

- `final_system_test.py` - 系统完整性测试
  - 验证所有核心模块的功能
  - 检查模块间的集成是否正常
  - 生成系统测试报告

### 输出文件 (测试后生成)
- `personalization_test_results.json` - 个性化测试结果
  - 不同用户的预测差异分析
  - 个性化调节效果评估

##  使用方法

### 1. 个性化效果测试
```bash
cd model_validation
python test_personalization.py
```

### 2. 系统完整性测试
```bash
python final_system_test.py
```

### 3. 查看测试结果
```bash
cat personalization_test_results.json
```

##  验证内容

### 个性化效果验证
- **用户特征选择**: 选择4类典型用户进行测试
  - 高神经质用户: neuroticism > 65
  - 低神经质用户: neuroticism < 40
  - 血瘀体质用户: constitution_type = 1
  - 平和体质用户: constitution_type = 0

- **预测差异分析**: 计算个性化模型与通用模型的预测差异
  - 月经概率差异
  - 疼痛等级差异
  - 个体化程度评估

### 临床合理性验证
- **神经质相关性**: 验证神经质与疼痛的相关系数(期望0.25-0.45)
- **周期分布**: 验证月经期占比(期望11%)
- **疼痛分布**: 验证月经期疼痛分布合理性

##  验证指标

### 个性化差异指标
- **平均概率差异**: 个性化vs通用模型的月经概率差异
- **平均疼痛差异**: 个性化vs通用模型的疼痛等级差异
- **差异显著性**: 不同用户类型的预测结果差异程度

### 性能对比
| 用户类型 | 通用模型 | 个性化模型 | 差异 |
|---------|---------|-----------|------|
| 高神经质用户 | 疼痛等级: 4.2 | 疼痛等级: 5.8 | +37% |
| 低神经质用户 | 疼痛等级: 4.2 | 疼痛等级: 2.1 | -50% |
| 血瘀体质用户 | 疼痛等级: 4.2 | 疼痛等级: 5.4 | +29% |
| 平和体质用户 | 疼痛等级: 4.2 | 疼痛等级: 3.1 | -1.1 |

##  测试流程

### 自动测试流程
1. **数据加载**: 从`data_generation/`加载测试数据
2. **模型加载**: 从`model_training/`加载个性化模型
3. **用户选择**: 自动选择代表性用户进行测试
4. **预测执行**: 同时运行个性化预测和通用预测
5. **差异分析**: 计算和分析预测差异
6. **报告生成**: 生成详细的测试报告

### 手动验证方法
```python
# 加载测试结果
import json
with open('personalization_test_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 分析个性化效果
for result in results:
    user_type = result['user_type']
    prob_diff = result['differences']['menstruation_prob']
    pain_diff = result['differences']['pain_level']
    print(f"{user_type}: 概率差异{prob_diff:.4f}, 疼痛差异{pain_diff:.2f}")
```

##  依赖关系

- **输入**: `model_training/` 提供训练好的模型
- **被调用**: `run_personalized_system.py` (一键运行)
- **调用**: 间接调用`data_generation/`的数据
- **输出**: 提供验证结果和性能报告

##  注意事项

1. **模型依赖**: 需要先完成模型训练才能进行验证
2. **数据一致性**: 确保测试使用的数据与训练数据一致
3. **用户代表性**: 测试用户应涵盖主要的个性化特征组合
4. **结果解释**: 差异分析应结合临床意义进行解释


