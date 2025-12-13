#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型预测脚本
用于加载训练好的模型进行预测
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import pickle
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class PersonalizedMultiTaskLSTM(nn.Module):
    """个性化多任务LSTM模型（与训练时保持一致）"""

    def __init__(self, input_size, user_feature_size=12, hidden_size=256, num_layers=3, dropout=0.25):
        super(PersonalizedMultiTaskLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 用户特征嵌入层
        self.user_embedding = nn.Sequential(
            nn.Linear(user_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )

        # 个性化特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 月经预测分支
        self.fc_menstruation = nn.Sequential(
            nn.Linear(hidden_size // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        # 疼痛等级预测分支
        self.fc_pain = nn.Sequential(
            nn.Linear(hidden_size // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # 个性化调节层
        self.personalization_layer = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4)
        )

    def forward(self, x, user_features=None):
        """个性化前向传播"""
        # 处理用户特征
        if user_features is not None:
            user_embedding = self.user_embedding(user_features)
        else:
            user_embedding = torch.zeros(x.size(0), 32, device=x.device)

        # LSTM前向传播
        lstm_out, _ = self.lstm(x)

        # 注意力机制
        attention_weights = self.attention(lstm_out)
        attention_output = torch.sum(lstm_out * attention_weights, dim=1)

        # 特征融合
        combined_features = torch.cat([attention_output, user_embedding], dim=1)
        fused_features = self.fusion_layer(combined_features)

        # 个性化调节
        personalization_adjust = self.personalization_layer(user_embedding)

        # 多任务输出
        menstruation_logits = self.fc_menstruation(fused_features)
        pain_pred = self.fc_pain(fused_features)

        # 应用个性化调节
        pain_adjust = personalization_adjust[:, 0:1] * 0.5
        pain_pred = pain_pred + pain_adjust

        return menstruation_logits, pain_pred.squeeze(-1)


class MultiTaskLSTM(nn.Module):
    """多任务LSTM模型（兼容旧版本）"""

    def __init__(self, input_size, hidden_size=384, num_layers=3, dropout=0.25):
        super(MultiTaskLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 共享特征提取层
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 月经预测分支（分类）
        self.fc_menstruation = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        # 疼痛等级预测分支（回归）
        self.fc_pain = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        shared_features = self.fc_shared(last_output)
        menstruation_logits = self.fc_menstruation(shared_features)
        pain_pred = self.fc_pain(shared_features)
        return menstruation_logits, pain_pred.squeeze(-1)


class DataPreprocessor:
    """数据预处理器（与训练时保持一致）"""
    
    def __init__(self):
        self.scaler = None
        self.phase_encoder = None
        self.feature_columns = None
        self.is_fitted = False
    
    def load_from_checkpoint(self, checkpoint):
        """从检查点加载预处理器"""
        # 注意：预处理器需要从训练数据中保存
        # 这里提供一个接口，实际使用时需要保存预处理器
        pass
    
    def transform(self, df: pd.DataFrame, scaler, phase_encoder, feature_columns):
        """转换数据"""
        df_processed = df.copy()
        df_processed['phase_encoded'] = phase_encoder.transform(df['phase'])
        
        X = df_processed[feature_columns].values
        X = scaler.transform(X)
        
        return X


class PersonalizedMenstrualCyclePredictor:
    """个性化月经周期预测器"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化个性化预测器

        Parameters:
        -----------
        model_path : str
            个性化模型文件路径
        device : str
            设备（'cpu'或'cuda'）
        """
        self.device = device
        self.model = None
        self.config = None
        self.scaler = None
        self.phase_encoder = None
        self.user_scaler = None
        self.feature_columns = None
        self.user_feature_columns = None
        self.window_size = 30

        self.load_personalized_model(model_path)

    def load_personalized_model(self, model_path: str):
        """加载个性化模型"""
        print(f"正在加载个性化模型: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # 加载配置
        self.config = checkpoint.get('config', {})
        self.window_size = self.config.get('window_size', 30)
        input_size = checkpoint.get('input_size', 9)
        user_feature_size = checkpoint.get('user_feature_size', 12)
        hidden_size = self.config.get('hidden_size', 256)
        num_layers = self.config.get('num_layers', 3)
        dropout = self.config.get('dropout', 0.25)

        # 创建个性化模型
        self.model = PersonalizedMultiTaskLSTM(
            input_size=input_size,
            user_feature_size=user_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print("✓ 个性化模型加载完成")

        # 加载个性化预处理器
        preprocessor_path = checkpoint.get('preprocessor_path', 'personalized_preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            self.scaler = preprocessor.scaler
            self.phase_encoder = preprocessor.phase_encoder
            self.user_scaler = preprocessor.user_scaler
            self.feature_columns = preprocessor.feature_columns
            self.user_feature_columns = preprocessor.user_feature_columns
            print("✓ 个性化预处理器加载完成")
        else:
            print("⚠️  警告: 个性化预处理器文件不存在，请确保personalized_preprocessor.pkl在同一目录")

    def predict(self, data: Dict, user_features: Dict) -> Dict:
        """
        进行个性化预测

        Parameters:
        -----------
        data : Dict
            时间序列数据字典
        user_features : Dict
            用户个性化特征字典

        Returns:
        --------
        Dict : 个性化预测结果
        """
        if self.scaler is None or self.user_scaler is None:
            raise ValueError("个性化预处理器未设置，请先调用set_personalized_preprocessor()")

        # 转换为DataFrame
        df = pd.DataFrame(data)
        user_df = pd.DataFrame([user_features])

        # 检查数据长度
        if len(df) < self.window_size:
            raise ValueError(f"数据长度不足，需要至少{self.window_size}天的数据")

        # 只取最后window_size天的数据
        df = df.tail(self.window_size).reset_index(drop=True)

        # 预处理
        X, X_user = self._preprocess_personalized(df, user_df)

        # 转换为序列格式
        X_seq = X.reshape(1, self.window_size, -1)
        X_user_seq = X_user.reshape(1, -1)

        # 预测
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            X_user_tensor = torch.FloatTensor(X_user_seq).to(self.device)
            menstruation_logits, pain_pred = self.model(X_tensor, X_user_tensor)

            menstruation_prob = torch.softmax(menstruation_logits, dim=1)[0, 1].item()
            pain_level = pain_pred[0].item()

        return {
            'menstruation_probability': max(0.0, min(1.0, menstruation_prob)),
            'pain_level': max(0.0, min(10.0, pain_level)),
            'is_menstruation': menstruation_prob > 0.5,
            'prediction_type': 'personalized'
        }

    def _preprocess_personalized(self, df: pd.DataFrame, user_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """个性化预处理"""
        df_processed = df.copy()
        df_processed['phase_encoded'] = self.phase_encoder.transform(df['phase'])

        X = df_processed[self.feature_columns].values
        X = self.scaler.transform(X)

        X_user = user_df[self.user_feature_columns].values
        X_user = self.user_scaler.transform(X_user)

        return X, X_user


class MenstrualCyclePredictor:
    """月经周期预测器（兼容旧版本）"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化预测器

        Parameters:
        -----------
        model_path : str
            模型文件路径
        device : str
            设备（'cpu'或'cuda'）
        """
        self.device = device
        self.model = None
        self.config = None
        self.scaler = None
        self.phase_encoder = None
        self.feature_columns = None
        self.window_size = 30

        self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载模型"""
        print(f"正在加载模型: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # 加载配置
        self.config = checkpoint.get('config', {})
        self.window_size = self.config.get('window_size', 30)
        input_size = checkpoint.get('input_size', 9)
        hidden_size = self.config.get('hidden_size', 384)
        num_layers = self.config.get('num_layers', 2)
        dropout = self.config.get('dropout', 0.3)

        # 创建模型
        self.model = MultiTaskLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print("✓ 模型加载完成")

        # 加载预处理器
        preprocessor_path = checkpoint.get('preprocessor_path', 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            self.scaler = preprocessor.scaler
            self.phase_encoder = preprocessor.phase_encoder
            self.feature_columns = preprocessor.feature_columns
            print("✓ 预处理器加载完成")
        else:
            print("⚠️  警告: 预处理器文件不存在，请确保preprocessor.pkl在同一目录")

    def set_preprocessor(self, scaler=None, phase_encoder=None, feature_columns=None):
        """设置预处理器（如果自动加载失败，可以手动设置）"""
        if scaler is not None:
            self.scaler = scaler
        if phase_encoder is not None:
            self.phase_encoder = phase_encoder
        if feature_columns is not None:
            self.feature_columns = feature_columns

    def predict(self, data: Dict) -> Dict:
        """
        进行预测（通用模式）

        Parameters:
        -----------
        data : Dict
            输入数据字典，包含以下键：
            - emotion: List[float] - 情绪得分列表（30天）
            - sleep_quality: List[float] - 睡眠质量列表（30天）
            - basal_body_temperature: List[float] - 基础体温列表（30天）
            - heart_rate: List[float] - 心率列表（30天）
            - stress_level: List[float] - 压力水平列表（30天）
            - disorder_score: List[float] - 紊乱度列表（30天）
            - cumulative_disorder: List[float] - 累积紊乱度列表（30天）
            - day_in_cycle: List[int] - 周期内天数列表（30天）
            - phase: List[str] - 周期阶段列表（30天）

        Returns:
        --------
        Dict : 预测结果
            - menstruation_probability: float - 月经概率
            - pain_level: float - 预测疼痛等级
        """
        if self.scaler is None or self.phase_encoder is None:
            raise ValueError("预处理器未设置，请先调用set_preprocessor()")

        # 转换为DataFrame
        df = pd.DataFrame(data)

        # 检查数据长度
        if len(df) < self.window_size:
            raise ValueError(f"数据长度不足，需要至少{self.window_size}天的数据")

        # 只取最后window_size天的数据
        df = df.tail(self.window_size).reset_index(drop=True)

        # 预处理
        X = self._preprocess(df)

        # 转换为序列格式
        X_seq = X.reshape(1, self.window_size, -1)

        # 预测
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            menstruation_logits, pain_pred = self.model(X_tensor)

            menstruation_prob = torch.softmax(menstruation_logits, dim=1)[0, 1].item()
            pain_level = pain_pred[0].item()

        return {
            'menstruation_probability': max(0.0, min(1.0, menstruation_prob)),
            'pain_level': max(0.0, min(10.0, pain_level)),
            'is_menstruation': menstruation_prob > 0.5,
            'prediction_type': 'general'
        }
    
    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """预处理数据"""
        df_processed = df.copy()
        df_processed['phase_encoded'] = self.phase_encoder.transform(df['phase'])
        
        X = df_processed[self.feature_columns].values
        X = self.scaler.transform(X)
        
        return X


def example_usage():
    """使用示例"""
    print("=" * 80)
    print("模型预测示例")
    print("=" * 80)
    
    # 加载模型
    predictor = MenstrualCyclePredictor('model.pth')
    
    # 注意：实际使用时需要加载预处理器
    # 这里只是示例
    print("\n⚠️  注意: 此示例需要配置预处理器")
    print("   在实际使用中，预处理器应该从训练时保存的文件中加载")
    
    # 示例数据（30天）
    example_data = {
        'emotion': np.random.rand(30) * 100,
        'sleep_quality': np.random.rand(30) * 100,
        'basal_body_temperature': np.random.rand(30) * 2 + 36,
        'heart_rate': np.random.rand(30) * 30 + 60,
        'stress_level': np.random.rand(30) * 100,
        'disorder_score': np.random.rand(30) * 10,
        'cumulative_disorder': np.random.rand(30) * 50,
        'day_in_cycle': np.random.randint(1, 31, 30),
        'phase': np.random.choice(['menstruation', 'follicular', 'ovulation', 'luteal'], 30)
    }
    
    print("\n示例数据已准备（随机生成）")
    print("实际使用时，请替换为真实的历史数据")


if __name__ == '__main__':
    example_usage()

