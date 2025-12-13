#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM模型 - 女性健康管理时间序列预测
支持多任务学习：月经时间预测（分类）+ 疼痛等级预测（回归）
"""

import numpy as np
import pandas as pd

# PyTorch导入检查
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except OSError as e:
    if "DLL" in str(e) or "c10.dll" in str(e):
        print("=" * 80)
        print("❌ PyTorch DLL加载错误")
        print("=" * 80)
        print("\n错误信息:")
        print(f"  {e}")
        print("\n解决方案:")
        print("  1. 运行修复脚本: fix_pytorch.bat")
        print("  2. 或手动执行以下命令:")
        print("     pip uninstall torch torchvision torchaudio -y")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print("  3. 安装 Visual C++ Redistributable:")
        print("     https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("\n详细说明请参考: Windows_PyTorch修复指南.md")
        print("=" * 80)
        exit(1)
    else:
        raise
except ImportError as e:
    print("=" * 80)
    print("❌ PyTorch未安装")
    print("=" * 80)
    print("\n请安装PyTorch:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("=" * 80)
    exit(1)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
import os
import pickle
import gc
from typing import Tuple, Dict, List
import warnings
import logging
from datetime import datetime
import time
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("警告: 未安装tqdm，将使用简单进度显示。建议安装: pip install tqdm")
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def setup_logger(log_file='training.log'):
    """
    配置日志记录器
    
    Parameters:
    -----------
    log_file : str
        日志文件路径
    
    Returns:
    --------
    logger : logging.Logger
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger('LSTM_Training')
    logger.setLevel(logging.INFO)
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 初始化日志记录器
logger = setup_logger()


class PersonalizedTimeSeriesDataset(Dataset):
    """个性化时间序列数据集类"""

    def __init__(self, X, X_user, y_menstruation, y_pain):
        self.X = torch.FloatTensor(X)
        self.X_user = torch.FloatTensor(X_user)
        self.y_menstruation = torch.LongTensor(y_menstruation)
        self.y_pain = torch.FloatTensor(y_pain)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_user[idx], self.y_menstruation[idx], self.y_pain[idx]


class TimeSeriesDataset(Dataset):
    """时间序列数据集类（兼容旧版本）"""

    def __init__(self, X, y_menstruation, y_pain):
        self.X = torch.FloatTensor(X)
        self.y_menstruation = torch.LongTensor(y_menstruation)
        self.y_pain = torch.FloatTensor(y_pain)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_menstruation[idx], self.y_pain[idx]


class PersonalizedMultiTaskLSTM(nn.Module):
    """个性化多任务LSTM模型：基于用户特征的个性化预测"""

    def __init__(self, input_size, user_feature_size=12, hidden_size=256, num_layers=3, dropout=0.3):
        super(PersonalizedMultiTaskLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 用户特征嵌入层 - 将用户个性化指标转换为向量
        self.user_embedding = nn.Sequential(
            nn.Linear(user_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM层 - 输入包含时间序列特征
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 注意力机制 - 关注重要时间步
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )

        # 个性化特征融合层 - 将LSTM输出与用户特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden_size),  # LSTM输出 + 用户嵌入
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 月经预测分支（个性化分类）
        self.fc_menstruation = nn.Sequential(
            nn.Linear(hidden_size // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 二分类：是否在月经期
        )

        # 疼痛等级预测分支（个性化回归）
        self.fc_pain = nn.Sequential(
            nn.Linear(hidden_size // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # 回归：疼痛等级（0-10分）
        )

        # 个性化调节层 - 基于临床研究的神经质影响
        self.personalization_layer = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 6)  # 调整6个输出参数：疼痛基线、敏感度、情绪影响、压力放大、周期调节、波动范围
        )


class MultiTaskLSTM(nn.Module):
    """兼容旧版本的多任务LSTM模型"""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
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
            nn.Linear(64, 2)  # 二分类：是否在月经期
        )

        # 疼痛等级预测分支（回归）
        self.fc_pain = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # 回归：疼痛等级（0-10分）
        )

    def forward(self, x, user_features=None):
        """个性化前向传播"""
        # 处理用户特征
        if user_features is not None:
            user_embedding = self.user_embedding(user_features)
        else:
            # 如果没有用户特征，使用零向量（通用模式）
            user_embedding = torch.zeros(x.size(0), 32, device=x.device)

        # LSTM前向传播
        lstm_out, _ = self.lstm(x)

        # 注意力机制 - 计算每个时间步的重要性
        attention_weights = self.attention(lstm_out)
        attention_output = torch.sum(lstm_out * attention_weights, dim=1)

        # 特征融合：LSTM输出 + 用户特征嵌入
        combined_features = torch.cat([attention_output, user_embedding], dim=1)
        fused_features = self.fusion_layer(combined_features)

        # 个性化调节 - 基于临床研究的精确调整
        personalization_params = self.personalization_layer(user_embedding)

        # 多任务输出
        menstruation_logits = self.fc_menstruation(fused_features)
        pain_pred = self.fc_pain(fused_features)

        # === 应用个性化调节参数 ===

        # 1. 疼痛基线调整（±1.0）
        pain_baseline_adjust = personalization_params[:, 0:1] * 1.0

        # 2. 疼痛敏感度调整（±0.5）
        pain_sensitivity_adjust = personalization_params[:, 1:2] * 0.5

        # 3. 情绪影响放大（0-2倍）
        emotion_amplifier = torch.sigmoid(personalization_params[:, 2:3]) * 2.0

        # 4. 压力响应调整（±0.8）
        stress_response_adjust = personalization_params[:, 3:4] * 0.8

        # 5. 周期阶段调节（±0.6）
        cycle_phase_adjust = personalization_params[:, 4:5] * 0.6

        # 6. 预测波动范围调整（0.1-0.5）
        prediction_variance = torch.sigmoid(personalization_params[:, 5:6]) * 0.4 + 0.1

        # === 综合个性化调整 ===

        # 应用基线和敏感度调整
        adjusted_pain_pred = pain_pred + pain_baseline_adjust + pain_sensitivity_adjust

        # 添加随机波动（考虑个性化波动范围）
        noise_scale = prediction_variance.expand_as(adjusted_pain_pred)
        personalized_noise = torch.randn_like(adjusted_pain_pred) * noise_scale
        adjusted_pain_pred = adjusted_pain_pred + personalized_noise

        # 确保预测在合理范围内
        adjusted_pain_pred = torch.clamp(adjusted_pain_pred, 0.0, 10.0)

        return menstruation_logits, adjusted_pain_pred.squeeze(-1)

    def forward_legacy(self, x):
        """兼容旧版本的forward方法"""
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 共享特征提取
        shared_features = self.fc_shared(last_output)

        # 多任务输出
        menstruation_logits = self.fc_menstruation(shared_features)
        pain_pred = self.fc_pain(shared_features)

        return menstruation_logits, pain_pred.squeeze(-1)


class PersonalizedDataPreprocessor:
    """个性化数据预处理器 - 支持用户特征"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.phase_encoder = LabelEncoder()
        self.user_scaler = StandardScaler()  # 用户特征标准化器
        self.feature_columns = None
        self.user_feature_columns = None
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame, user_attrs_df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        拟合并转换数据（包含用户特征）

        Parameters:
        -----------
        df : pd.DataFrame
            时间序列数据框
        user_attrs_df : pd.DataFrame
            用户属性数据框

        Returns:
        --------
        X : np.ndarray
            时间序列特征数组 (n_samples, n_features)
        X_user : np.ndarray
            用户特征数组 (n_samples, user_feature_size)
        y_menstruation : np.ndarray
            月经标签 (n_samples,)
        y_pain : np.ndarray
            疼痛等级标签 (n_samples,)
        """
        # 时间序列特征列
        self.feature_columns = [
            'emotion', 'sleep_quality', 'basal_body_temperature',
            'heart_rate', 'stress_level', 'disorder_score',
            'cumulative_disorder', 'day_in_cycle'
        ]

        # 用户特征列
        self.user_feature_columns = [
            'cycle_length', 'neuroticism', 'trait_anxiety', 'psychoticism',
            'constitution_type', 'constitution_coef', 'is_night_owl',
            'base_sleep_quality', 'base_emotion', 'base_heart_rate',
            'base_pain_level', 'stress_sensitivity'
        ]

        # 处理phase特征（编码）
        df_processed = df.copy()
        df_processed['phase_encoded'] = self.phase_encoder.fit_transform(df['phase'])
        self.feature_columns.append('phase_encoded')

        # 提取时间序列特征
        X = df_processed[self.feature_columns].values
        X = self.scaler.fit_transform(X)

        # 提取用户特征
        X_user = np.zeros((len(df), len(self.user_feature_columns)))
        if user_attrs_df is not None:
            for idx, row in df.iterrows():
                user_id = int(row['user_id'])
                user_data = user_attrs_df[user_attrs_df['user_id'] == user_id]
                if not user_data.empty:
                    user_features = user_data[self.user_feature_columns].values[0]
                    X_user[idx] = user_features

        # 标准化用户特征
        X_user = self.user_scaler.fit_transform(X_user)

        # 提取标签
        y_menstruation = df['menstruation'].values
        y_pain = df['pain_level'].values

        self.is_fitted = True

        return X, X_user, y_menstruation, y_pain

    def transform(self, df: pd.DataFrame, user_attrs_df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """转换新数据（使用已拟合的scaler和encoder）"""
        if not self.is_fitted:
            raise ValueError("预处理器尚未拟合，请先调用fit_transform")

        df_processed = df.copy()
        df_processed['phase_encoded'] = self.phase_encoder.transform(df['phase'])

        # 时间序列特征
        X = df_processed[self.feature_columns].values
        X = self.scaler.transform(X)

        # 用户特征
        X_user = np.zeros((len(df), len(self.user_feature_columns)))
        if user_attrs_df is not None:
            for idx, row in df.iterrows():
                user_id = int(row['user_id'])
                user_data = user_attrs_df[user_attrs_df['user_id'] == user_id]
                if not user_data.empty:
                    user_features = user_data[self.user_feature_columns].values[0]
                    X_user[idx] = user_features

        X_user = self.user_scaler.transform(X_user)

        y_menstruation = df['menstruation'].values
        y_pain = df['pain_level'].values

        return X, X_user, y_menstruation, y_pain


class DataPreprocessor:
    """兼容旧版本的数据预处理器"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.phase_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        拟合并转换数据

        Parameters:
        -----------
        df : pd.DataFrame
            原始数据框

        Returns:
        --------
        X : np.ndarray
            特征数组 (n_samples, n_features)
        y_menstruation : np.ndarray
            月经标签 (n_samples,)
        y_pain : np.ndarray
            疼痛等级标签 (n_samples,)
        """
        # 选择特征列
        self.feature_columns = [
            'emotion', 'sleep_quality', 'basal_body_temperature',
            'heart_rate', 'stress_level', 'disorder_score',
            'cumulative_disorder', 'day_in_cycle'
        ]

        # 处理phase特征（编码）
        df_processed = df.copy()
        df_processed['phase_encoded'] = self.phase_encoder.fit_transform(df['phase'])
        self.feature_columns.append('phase_encoded')

        # 提取特征
        X = df_processed[self.feature_columns].values

        # 标准化特征
        X = self.scaler.fit_transform(X)

        # 提取标签
        y_menstruation = df['menstruation'].values
        y_pain = df['pain_level'].values

        self.is_fitted = True

        return X, y_menstruation, y_pain

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """转换新数据（使用已拟合的scaler和encoder）"""
        if not self.is_fitted:
            raise ValueError("预处理器尚未拟合，请先调用fit_transform")

        df_processed = df.copy()
        df_processed['phase_encoded'] = self.phase_encoder.transform(df['phase'])

        X = df_processed[self.feature_columns].values
        X = self.scaler.transform(X)

        y_menstruation = df['menstruation'].values
        y_pain = df['pain_level'].values

        return X, y_menstruation, y_pain


def create_sequences(X, y_menstruation, y_pain, window_size=30):
    """
    创建时间序列窗口
    
    Parameters:
    -----------
    X : np.ndarray
        特征数组 (n_samples, n_features)
    y_menstruation : np.ndarray
        月经标签
    y_pain : np.ndarray
        疼痛等级标签
    window_size : int
        窗口大小（天数）
    
    Returns:
    --------
    X_seq : np.ndarray
        序列特征 (n_sequences, window_size, n_features)
    y_menstruation_seq : np.ndarray
        序列标签（月经）
    y_pain_seq : np.ndarray
        序列标签（疼痛）
    """
    X_seq = []
    y_menstruation_seq = []
    y_pain_seq = []
    
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i+window_size])
        y_menstruation_seq.append(y_menstruation[i+window_size-1])
        y_pain_seq.append(y_pain[i+window_size-1])
    
    return np.array(X_seq), np.array(y_menstruation_seq), np.array(y_pain_seq)


def load_and_preprocess_data_personalized(data_path: str, user_attrs_path: str = None,
                                         window_size: int = 30, test_size: float = 0.2,
                                         val_size: float = 0.1) -> Dict:
    """
    加载和预处理个性化数据（包含用户特征）

    Parameters:
    -----------
    data_path : str
        时间序列数据文件路径
    user_attrs_path : str
        用户属性数据文件路径
    window_size : int
        时间窗口大小
    test_size : float
        测试集比例
    val_size : float
        验证集比例（从训练集中划分）

    Returns:
    --------
    Dict : 包含预处理后的数据和预处理器
    """
    start_time = time.time()
    step_start = time.time()

    logger.info("=" * 80)
    logger.info("步骤1: 加载个性化数据")
    logger.info("=" * 80)
    print(f"\n[步骤1/4] 正在加载个性化数据: {data_path}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载数据
    df = pd.read_csv(data_path)
    user_attrs_df = None
    if user_attrs_path and os.path.exists(user_attrs_path):
        user_attrs_df = pd.read_csv(user_attrs_path)
        print(f"  ✓ 用户属性数据已加载: {len(user_attrs_df)} 个用户")

    step_time = time.time() - step_start
    logger.info(f"个性化数据加载完成: {len(df):,} 条记录, 耗时: {step_time:.2f}秒")
    print(f"  ✓ 个性化数据加载完成: {len(df):,} 条记录 (耗时: {step_time:.2f}秒)")

    # 使用个性化预处理器
    step_start = time.time()
    logger.info("=" * 80)
    logger.info("步骤2: 拟合个性化预处理器")
    logger.info("=" * 80)
    print(f"\n[步骤2/4] 正在拟合个性化预处理器...")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    preprocessor = PersonalizedDataPreprocessor()
    preprocessor.fit_transform(df, user_attrs_df)  # 只用于拟合

    step_time = time.time() - step_start
    logger.info(f"个性化预处理器拟合完成, 耗时: {step_time:.2f}秒")
    print(f"  ✓ 个性化预处理器拟合完成 (耗时: {step_time:.2f}秒)")

    # 按用户分组，为每个用户创建序列
    step_start = time.time()
    logger.info("=" * 80)
    logger.info("步骤3: 创建个性化时间序列窗口")
    logger.info("=" * 80)
    print(f"\n[步骤3/4] 正在创建个性化时间序列窗口...")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_X_seq = []
    all_X_user_seq = []
    all_y_menstruation_seq = []
    all_y_pain_seq = []

    user_ids = df['user_id'].unique()
    total_users = len(user_ids)
    user_count = 0
    skipped_users = 0

    # 使用进度条
    if HAS_TQDM:
        user_iterator = tqdm(user_ids, desc="处理用户", unit="用户")
    else:
        user_iterator = user_ids

    for user_id in user_iterator:
        user_df = df[df['user_id'] == user_id].sort_values('date').reset_index(drop=True)

        if len(user_df) < window_size:
            skipped_users += 1
            continue

        # 使用已拟合的预处理器转换用户数据
        X, X_user, y_menstruation, y_pain = preprocessor.transform(user_df, user_attrs_df)

        # 创建序列
        X_seq, y_menstruation_seq, y_pain_seq = create_sequences(X, y_menstruation, y_pain, window_size)
        X_user_seq = np.repeat([X_user[window_size-1]], len(X_seq), axis=0)  # 用户特征对每个序列都一样

        all_X_seq.append(X_seq)
        all_X_user_seq.append(X_user_seq)
        all_y_menstruation_seq.append(y_menstruation_seq)
        all_y_pain_seq.append(y_pain_seq)
        user_count += 1

        if not HAS_TQDM and user_count % max(1, total_users // 20) == 0:
            progress = user_count / total_users * 100
            print(f"  进度: {user_count}/{total_users} 用户 ({progress:.1f}%)")
            logger.info(f"已处理 {user_count}/{total_users} 个用户 ({progress:.1f}%)...")

    step_time = time.time() - step_start
    logger.info(f"个性化用户处理完成: 有效用户 {user_count} 个, 跳过 {skipped_users} 个，耗时: {step_time:.2f}秒")
    print(f"  ✓ 个性化用户处理完成: 有效用户 {user_count} 个, 跳过 {skipped_users} 个 (耗时: {step_time:.2f}秒)")

    # 合并所有用户的数据
    step_start = time.time()
    logger.info("正在合并所有个性化用户数据...")
    print("  正在合并个性化数据...")
    print(f"    需要合并 {len(all_X_seq)} 个用户的数据...")

    # 对于大数据集，分批合并以避免内存问题
    if len(all_X_seq) > 100:
        print(f"    数据量较大，使用分批合并...")
        batch_size = 100
        X_batches = []
        X_user_batches = []
        y_m_batches = []
        y_p_batches = []

        for i in range(0, len(all_X_seq), batch_size):
            batch_X = all_X_seq[i:i+batch_size]
            batch_X_user = all_X_user_seq[i:i+batch_size]
            batch_y_m = all_y_menstruation_seq[i:i+batch_size]
            batch_y_p = all_y_pain_seq[i:i+batch_size]

            X_batches.append(np.concatenate(batch_X, axis=0))
            X_user_batches.append(np.concatenate(batch_X_user, axis=0))
            y_m_batches.append(np.concatenate(batch_y_m, axis=0))
            y_p_batches.append(np.concatenate(batch_y_p, axis=0))

            if (i // batch_size + 1) % 10 == 0:
                print(f"      已合并 {min(i+batch_size, len(all_X_seq))}/{len(all_X_seq)} 个用户...")

        X_all = np.concatenate(X_batches, axis=0)
        X_user_all = np.concatenate(X_user_batches, axis=0)
        y_menstruation_all = np.concatenate(y_m_batches, axis=0)
        y_pain_all = np.concatenate(y_p_batches, axis=0)
    else:
        X_all = np.concatenate(all_X_seq, axis=0)
        X_user_all = np.concatenate(all_X_user_seq, axis=0)
        y_menstruation_all = np.concatenate(all_y_menstruation_seq, axis=0)
        y_pain_all = np.concatenate(all_y_pain_seq, axis=0)

    merge_time = time.time() - step_start
    logger.info(f"个性化序列创建完成: {len(X_all):,} 个样本, 合并耗时: {merge_time:.2f}秒")
    print(f"  ✓ 个性化序列创建完成: {len(X_all):,} 个样本 (合并耗时: {merge_time:.2f}秒)")
    print(f"    时间序列特征维度: {X_all.shape}")
    print(f"    用户特征维度: {X_user_all.shape}")

    # 划分训练集和测试集
    step_start = time.time()
    logger.info("=" * 80)
    logger.info("步骤4: 划分个性化数据集")
    logger.info("=" * 80)
    print(f"\n[步骤4/4] 正在划分个性化数据集...")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据量: {len(X_all):,} 个样本")

    # 计算原始分布
    original_dist = np.bincount(y_menstruation_all)
    original_rate = original_dist[1] / len(y_menstruation_all) if len(original_dist) > 1 else 0
    print(f"  原始数据分布: 非月经期 {original_dist[0]:,} ({original_dist[0]/len(y_menstruation_all)*100:.2f}%), "
          f"月经期 {original_dist[1]:,} ({original_rate*100:.2f}%)")

    print("  使用分层抽样（stratify）确保分布完全一致（优先准确性）...")
    logger.info("使用分层抽样划分个性化数据集（硬件配置：128GB内存，可支持完整stratify）")

    # 保存需要的信息
    total_samples = len(X_all)
    input_size = X_all.shape[2]
    user_feature_size = X_user_all.shape[1]

    # 第一次划分：训练+验证 vs 测试
    print("  第一步：划分训练+验证集 vs 测试集...")
    X_train_val, X_test, X_user_train_val, X_user_test, y_menstruation_train_val, y_menstruation_test, y_pain_train_val, y_pain_test = train_test_split(
        X_all, X_user_all, y_menstruation_all, y_pain_all,
        test_size=test_size, random_state=42, stratify=y_menstruation_all
    )

    # 释放原始数据内存
    del X_all, X_user_all, y_menstruation_all, y_pain_all
    gc.collect()
    print("  ✓ 第一步完成，已释放原始数据内存")

    # 第二次划分：训练 vs 验证
    print("  第二步：划分训练集 vs 验证集...")
    X_train, X_val, X_user_train, X_user_val, y_menstruation_train, y_menstruation_val, y_pain_train, y_pain_val = train_test_split(
        X_train_val, X_user_train_val, y_menstruation_train_val, y_pain_train_val,
        test_size=val_size / (1 - test_size), random_state=42, stratify=y_menstruation_train_val
    )

    # 释放中间数据
    del X_train_val, X_user_train_val, y_menstruation_train_val, y_pain_train_val
    gc.collect()
    print("  ✓ 第二步完成，数据划分完成")

    # 验证分布一致性
    train_dist = np.bincount(y_menstruation_train)
    val_dist = np.bincount(y_menstruation_val)
    test_dist = np.bincount(y_menstruation_test)

    train_rate = train_dist[1] / len(y_menstruation_train) if len(train_dist) > 1 else 0
    val_rate = val_dist[1] / len(y_menstruation_val) if len(val_dist) > 1 else 0
    test_rate = test_dist[1] / len(y_menstruation_test) if len(test_dist) > 1 else 0

    print(f"  分布验证（应完全一致）:")
    print(f"    训练集: {train_rate*100:.4f}% (原始: {original_rate*100:.4f}%)")
    print(f"    验证集: {val_rate*100:.4f}% (原始: {original_rate*100:.4f}%)")
    print(f"    测试集: {test_rate*100:.4f}% (原始: {original_rate*100:.4f}%)")
    logger.info(f"分布验证: 训练 {train_rate*100:.4f}%, 验证 {val_rate*100:.4f}%, 测试 {test_rate*100:.4f}% (原始 {original_rate*100:.4f}%)")

    if abs(train_rate - original_rate) < 0.0001 and abs(val_rate - original_rate) < 0.0001 and abs(test_rate - original_rate) < 0.0001:
        print("  ✓ 分布完全一致，分层抽样成功")
    else:
        logger.warning("分布略有偏差，但仍在可接受范围内")

    print("  ✓ 数据划分完成")

    step_time = time.time() - step_start
    total_time = time.time() - start_time
    logger.info(f"个性化数据划分完成, 耗时: {step_time:.2f}秒")
    logger.info(f"个性化数据预处理总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"  ✓ 数据划分完成 (耗时: {step_time:.2f}秒)")
    print(f"\n个性化数据预处理总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"\n个性化数据划分结果:")
    print(f"  训练集: {len(X_train):,} 个样本 ({len(X_train)/total_samples*100:.1f}%)")
    print(f"  验证集: {len(X_val):,} 个样本 ({len(X_val)/total_samples*100:.1f}%)")
    print(f"  测试集: {len(X_test):,} 个样本 ({len(X_test)/total_samples*100:.1f}%)")

    return {
        'X_train': X_train, 'X_user_train': X_user_train, 'y_menstruation_train': y_menstruation_train, 'y_pain_train': y_pain_train,
        'X_val': X_val, 'X_user_val': X_user_val, 'y_menstruation_val': y_menstruation_val, 'y_pain_val': y_pain_val,
        'X_test': X_test, 'X_user_test': X_user_test, 'y_menstruation_test': y_menstruation_test, 'y_pain_test': y_pain_test,
        'preprocessor': preprocessor,
        'input_size': input_size,
        'user_feature_size': user_feature_size
    }


def load_and_preprocess_data(data_path: str, window_size: int = 30,
                             test_size: float = 0.2, val_size: float = 0.1) -> Dict:
    """
    加载和预处理数据
    
    Parameters:
    -----------
    data_path : str
        数据文件路径
    window_size : int
        时间窗口大小
    test_size : float
        测试集比例
    val_size : float
        验证集比例（从训练集中划分）
    
    Returns:
    --------
    Dict : 包含预处理后的数据和预处理器
    """
    start_time = time.time()
    step_start = time.time()
    
    logger.info("=" * 80)
    logger.info("步骤1: 加载数据")
    logger.info("=" * 80)
    print(f"\n[步骤1/4] 正在加载数据: {data_path}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    df = pd.read_csv(data_path)
    step_time = time.time() - step_start
    logger.info(f"数据加载完成: {len(df):,} 条记录, 耗时: {step_time:.2f}秒")
    logger.info(f"数据列: {list(df.columns)}")
    print(f"  ✓ 数据加载完成: {len(df):,} 条记录 (耗时: {step_time:.2f}秒)")
    
    # 先对所有数据拟合预处理器
    step_start = time.time()
    logger.info("=" * 80)
    logger.info("步骤2: 拟合预处理器")
    logger.info("=" * 80)
    print(f"\n[步骤2/4] 正在拟合预处理器...")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    preprocessor = DataPreprocessor()
    preprocessor.fit_transform(df)  # 只用于拟合，不使用返回值
    
    step_time = time.time() - step_start
    logger.info(f"预处理器拟合完成, 耗时: {step_time:.2f}秒")
    print(f"  ✓ 预处理器拟合完成 (耗时: {step_time:.2f}秒)")
    
    # 按用户分组，为每个用户创建序列
    step_start = time.time()
    logger.info("=" * 80)
    logger.info("步骤3: 创建时间序列窗口")
    logger.info("=" * 80)
    print(f"\n[步骤3/4] 正在创建时间序列窗口...")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_X_seq = []
    all_y_menstruation_seq = []
    all_y_pain_seq = []
    
    user_ids = df['user_id'].unique()
    total_users = len(user_ids)
    user_count = 0
    skipped_users = 0
    
    # 使用进度条
    if HAS_TQDM:
        user_iterator = tqdm(user_ids, desc="处理用户", unit="用户")
    else:
        user_iterator = user_ids
    
    for user_id in user_iterator:
        user_df = df[df['user_id'] == user_id].sort_values('date').reset_index(drop=True)
        
        if len(user_df) < window_size:
            skipped_users += 1
            continue
        
        # 使用已拟合的预处理器转换用户数据
        X, y_menstruation, y_pain = preprocessor.transform(user_df)
        
        # 创建序列
        X_seq, y_menstruation_seq, y_pain_seq = create_sequences(
            X, y_menstruation, y_pain, window_size
        )
        
        all_X_seq.append(X_seq)
        all_y_menstruation_seq.append(y_menstruation_seq)
        all_y_pain_seq.append(y_pain_seq)
        user_count += 1
        
        if not HAS_TQDM and user_count % max(1, total_users // 20) == 0:
            progress = user_count / total_users * 100
            print(f"  进度: {user_count}/{total_users} 用户 ({progress:.1f}%)")
            logger.info(f"已处理 {user_count}/{total_users} 个用户 ({progress:.1f}%)...")
    
    step_time = time.time() - step_start
    logger.info(f"用户处理完成: 有效用户 {user_count} 个, 跳过 {skipped_users} 个（数据不足）, 耗时: {step_time:.2f}秒")
    print(f"  ✓ 用户处理完成: 有效用户 {user_count} 个, 跳过 {skipped_users} 个 (耗时: {step_time:.2f}秒)")
    
    # 合并所有用户的数据
    step_start = time.time()
    logger.info("正在合并所有用户数据...")
    print("  正在合并数据...")
    print(f"    需要合并 {len(all_X_seq)} 个用户的数据...")
    
    # 对于大数据集，分批合并以避免内存问题
    if len(all_X_seq) > 100:
        print(f"    数据量较大，使用分批合并...")
        batch_size = 100
        X_batches = []
        y_m_batches = []
        y_p_batches = []
        
        for i in range(0, len(all_X_seq), batch_size):
            batch_X = all_X_seq[i:i+batch_size]
            batch_y_m = all_y_menstruation_seq[i:i+batch_size]
            batch_y_p = all_y_pain_seq[i:i+batch_size]
            
            X_batches.append(np.concatenate(batch_X, axis=0))
            y_m_batches.append(np.concatenate(batch_y_m, axis=0))
            y_p_batches.append(np.concatenate(batch_y_p, axis=0))
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"      已合并 {min(i+batch_size, len(all_X_seq))}/{len(all_X_seq)} 个用户...")
        
        X_all = np.concatenate(X_batches, axis=0)
        y_menstruation_all = np.concatenate(y_m_batches, axis=0)
        y_pain_all = np.concatenate(y_p_batches, axis=0)
    else:
        X_all = np.concatenate(all_X_seq, axis=0)
        y_menstruation_all = np.concatenate(all_y_menstruation_seq, axis=0)
        y_pain_all = np.concatenate(all_y_pain_seq, axis=0)
    
    merge_time = time.time() - step_start
    logger.info(f"序列创建完成: {len(X_all):,} 个样本, 合并耗时: {merge_time:.2f}秒")
    logger.info(f"  特征维度: {X_all.shape}")
    logger.info(f"  月经标签分布: {np.bincount(y_menstruation_all)}")
    logger.info(f"  疼痛等级范围: {y_pain_all.min():.2f} - {y_pain_all.max():.2f}")
    print(f"  ✓ 序列创建完成: {len(X_all):,} 个样本 (合并耗时: {merge_time:.2f}秒)")
    print(f"    特征维度: {X_all.shape}")
    print(f"  月经标签分布: {np.bincount(y_menstruation_all)}")
    print(f"  疼痛等级范围: {y_pain_all.min():.2f} - {y_pain_all.max():.2f}")
    
    # 划分训练集和测试集
    step_start = time.time()
    logger.info("=" * 80)
    logger.info("步骤4: 划分数据集")
    logger.info("=" * 80)
    print(f"\n[步骤4/4] 正在划分数据集...")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据量: {len(X_all):,} 个样本")
    
    # 计算原始分布
    original_dist = np.bincount(y_menstruation_all)
    original_rate = original_dist[1] / len(y_menstruation_all) if len(original_dist) > 1 else 0
    print(f"  原始数据分布: 非月经期 {original_dist[0]:,} ({original_dist[0]/len(y_menstruation_all)*100:.2f}%), "
          f"月经期 {original_dist[1]:,} ({original_rate*100:.2f}%)")
    
    # 硬件配置：14900K/128GB内存 - 使用分层抽样确保准确性
    # 128GB内存足够处理691万样本，直接使用stratify
    print("  使用分层抽样（stratify）确保分布完全一致（优先准确性）...")
    logger.info("使用分层抽样划分数据集（硬件配置：128GB内存，可支持完整stratify）")
    
    # 保存需要的信息（在删除之前）
    total_samples = len(X_all)
    input_size = X_all.shape[2]
    
    # 第一次划分：训练+验证 vs 测试
    print("  第一步：划分训练+验证集 vs 测试集...")
    X_train_val, X_test, y_menstruation_train_val, y_menstruation_test, y_pain_train_val, y_pain_test = train_test_split(
        X_all, y_menstruation_all, y_pain_all, 
        test_size=test_size, random_state=42, stratify=y_menstruation_all
    )
    
    # 释放原始数据内存
    del X_all, y_menstruation_all, y_pain_all
    gc.collect()
    print("  ✓ 第一步完成，已释放原始数据内存")
    
    # 第二次划分：训练 vs 验证
    print("  第二步：划分训练集 vs 验证集...")
    X_train, X_val, y_menstruation_train, y_menstruation_val, y_pain_train, y_pain_val = train_test_split(
        X_train_val, y_menstruation_train_val, y_pain_train_val,
        test_size=val_size / (1 - test_size), random_state=42, stratify=y_menstruation_train_val
    )
    
    # 释放中间数据
    del X_train_val, y_menstruation_train_val, y_pain_train_val
    gc.collect()
    print("  ✓ 第二步完成，数据划分完成")
    
    # 验证分布一致性
    train_dist = np.bincount(y_menstruation_train)
    val_dist = np.bincount(y_menstruation_val)
    test_dist = np.bincount(y_menstruation_test)
    
    train_rate = train_dist[1] / len(y_menstruation_train) if len(train_dist) > 1 else 0
    val_rate = val_dist[1] / len(y_menstruation_val) if len(val_dist) > 1 else 0
    test_rate = test_dist[1] / len(y_menstruation_test) if len(test_dist) > 1 else 0
    
    print(f"  分布验证（应完全一致）:")
    print(f"    训练集: {train_rate*100:.4f}% (原始: {original_rate*100:.4f}%)")
    print(f"    验证集: {val_rate*100:.4f}% (原始: {original_rate*100:.4f}%)")
    print(f"    测试集: {test_rate*100:.4f}% (原始: {original_rate*100:.4f}%)")
    logger.info(f"分布验证: 训练 {train_rate*100:.4f}%, 验证 {val_rate*100:.4f}%, 测试 {test_rate*100:.4f}% (原始 {original_rate*100:.4f}%)")
    
    if abs(train_rate - original_rate) < 0.0001 and abs(val_rate - original_rate) < 0.0001 and abs(test_rate - original_rate) < 0.0001:
        print("  ✓ 分布完全一致，分层抽样成功")
    else:
        logger.warning("分布略有偏差，但仍在可接受范围内")
    
    
    print("  ✓ 数据划分完成")
    
    step_time = time.time() - step_start
    total_time = time.time() - start_time
    logger.info(f"数据划分完成, 耗时: {step_time:.2f}秒")
    logger.info(f"数据预处理总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"  ✓ 数据划分完成 (耗时: {step_time:.2f}秒)")
    print(f"\n数据预处理总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"\n数据划分结果:")
    print(f"  训练集: {len(X_train):,} 个样本 ({len(X_train)/total_samples*100:.1f}%)")
    print(f"  验证集: {len(X_val):,} 个样本 ({len(X_val)/total_samples*100:.1f}%)")
    print(f"  测试集: {len(X_test):,} 个样本 ({len(X_test)/total_samples*100:.1f}%)")
    
    return {
        'X_train': X_train, 'y_menstruation_train': y_menstruation_train, 'y_pain_train': y_pain_train,
        'X_val': X_val, 'y_menstruation_val': y_menstruation_val, 'y_pain_val': y_pain_val,
        'X_test': X_test, 'y_menstruation_test': y_menstruation_test, 'y_pain_test': y_pain_test,
        'preprocessor': preprocessor,
        'input_size': input_size
    }


def train_personalized_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001,
                            device='cpu', patience=10, gradient_accumulation_steps=1, warmup_epochs=0):
    """
    训练个性化模型

    Parameters:
    -----------
    model : nn.Module
        个性化LSTM模型
    train_loader : DataLoader
        训练数据加载器（包含用户特征）
    val_loader : DataLoader
        验证数据加载器（包含用户特征）
    num_epochs : int
        训练轮数
    learning_rate : float
        学习率
    device : str
        设备（'cpu'或'cuda'）
    patience : int
        早停耐心值
    gradient_accumulation_steps : int
        梯度累积步数
    warmup_epochs : int
        预热轮数

    Returns:
    --------
    Dict : 训练历史
    """
    model = model.to(device)

    # 损失函数
    criterion_menstruation = nn.CrossEntropyLoss()
    criterion_pain = nn.MSELoss()

    # 优化器 - 使用更强的权重衰减和AdamW
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))

    # 学习率调度器：预热 + 余弦退火
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 预热阶段：线性增加到初始学习率
            return (epoch + 1) / warmup_epochs
        else:
            # 余弦退火阶段
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_menstruation_acc': [], 'val_menstruation_acc': [],
        'train_pain_mae': [], 'val_pain_mae': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    train_start_time = time.time()
    logger.info("=" * 80)
    logger.info("开始训练个性化模型")
    logger.info("=" * 80)
    logger.info(f"设备: {device}")
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    logger.info(f"总训练轮数: {num_epochs}, 早停耐心值: {patience}")
    logger.info(f"梯度累积步数: {gradient_accumulation_steps}")
    logger.info(f"学习率预热轮数: {warmup_epochs}")
    print("\n" + "=" * 80)
    print("开始训练个性化模型")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
    print(f"总训练轮数: {num_epochs}, 早停耐心值: {patience}")
    print(f"梯度累积步数: {gradient_accumulation_steps}")
    print(f"学习率预热轮数: {warmup_epochs}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch+1}/{num_epochs} 开始")

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_menstruation_correct = 0
        train_menstruation_total = 0
        train_pain_errors = []

        # 使用tqdm显示训练进度（如果可用）
        if HAS_TQDM:
            train_iterator = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}", unit="批次",
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            train_iterator = train_loader

        # 梯度累积训练
        accumulation_counter = 0
        optimizer.zero_grad()  # 在循环外初始化梯度

        for X_batch, X_user_batch, y_menstruation_batch, y_pain_batch in train_iterator:
            X_batch = X_batch.to(device)
            X_user_batch = X_user_batch.to(device)
            y_menstruation_batch = y_menstruation_batch.to(device)
            y_pain_batch = y_pain_batch.to(device)

            # 前向传播（个性化）
            menstruation_logits, pain_pred = model(X_batch, X_user_batch)

            # 计算损失（多任务损失加权）
            loss_menstruation = criterion_menstruation(menstruation_logits, y_menstruation_batch)
            loss_pain = criterion_pain(pain_pred, y_pain_batch)
            loss = (loss_menstruation + 0.5 * loss_pain) / gradient_accumulation_steps  # 归一化损失

            # 反向传播（累积梯度）
            loss.backward()

            accumulation_counter += 1

            # 达到累积步数时更新参数
            if accumulation_counter % gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 统计
            train_loss += loss.item()
            train_menstruation_correct += (menstruation_logits.argmax(1) == y_menstruation_batch).sum().item()
            train_menstruation_total += y_menstruation_batch.size(0)
            train_pain_errors.extend((pain_pred - y_pain_batch).abs().cpu().numpy())

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_menstruation_correct = 0
        val_menstruation_total = 0
        val_pain_errors = []

        with torch.no_grad():
            for X_batch, X_user_batch, y_menstruation_batch, y_pain_batch in val_loader:
                X_batch = X_batch.to(device)
                X_user_batch = X_user_batch.to(device)
                y_menstruation_batch = y_menstruation_batch.to(device)
                y_pain_batch = y_pain_batch.to(device)

                menstruation_logits, pain_pred = model(X_batch, X_user_batch)

                loss_menstruation = criterion_menstruation(menstruation_logits, y_menstruation_batch)
                loss_pain = criterion_pain(pain_pred, y_pain_batch)
                loss = loss_menstruation + 0.5 * loss_pain

                val_loss += loss.item()
                val_menstruation_correct += (menstruation_logits.argmax(1) == y_menstruation_batch).sum().item()
                val_menstruation_total += y_menstruation_batch.size(0)
                val_pain_errors.extend((pain_pred - y_pain_batch).abs().cpu().numpy())

        # 计算平均指标
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_menstruation_correct / train_menstruation_total
        val_acc = val_menstruation_correct / val_menstruation_total
        train_mae = np.mean(train_pain_errors)
        val_mae = np.mean(val_pain_errors)

        # 更新学习率 (LambdaLR 需要传入当前epoch)
        scheduler.step(epoch)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_menstruation_acc'].append(train_acc)
        history['val_menstruation_acc'].append(val_acc)
        history['train_pain_mae'].append(train_mae)
        history['val_pain_mae'].append(val_mae)

        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log_msg = (f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"训练 Loss: {train_loss:.4f}, 月经准确率: {train_acc:.4f}, 疼痛MAE: {train_mae:.4f} | "
                      f"验证 Loss: {val_loss:.4f}, 月经准确率: {val_acc:.4f}, 疼痛MAE: {val_mae:.4f} | "
                      f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
            logger.info(log_msg)
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  训练 - Loss: {train_loss:.4f}, 月经准确率: {train_acc:.4f}, 疼痛MAE: {train_mae:.4f}")
            print(f"  验证 - Loss: {val_loss:.4f}, 月经准确率: {val_acc:.4f}, 疼痛MAE: {val_mae:.4f}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_personalized_model.pth')
            logger.info(f"Epoch [{epoch+1}] - 发现更好的模型，验证损失: {val_loss:.4f}，已保存个性化模型")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.warning(f"早停触发！在第 {epoch+1} 轮停止训练（验证损失连续 {patience} 轮未改善）")
                print(f"\n早停触发！在第 {epoch+1} 轮停止训练")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_personalized_model.pth'))
    logger.info(f"个性化训练完成！最佳验证损失: {best_val_loss:.4f}")
    print("\n个性化训练完成！")

    return history


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001,
                device='cpu', patience=10, gradient_accumulation_steps=1, warmup_epochs=0):
    """
    训练模型
    
    Parameters:
    -----------
    model : nn.Module
        LSTM模型
    train_loader : DataLoader
        训练数据加载器
    val_loader : DataLoader
        验证数据加载器
    num_epochs : int
        训练轮数
    learning_rate : float
        学习率
    device : str
        设备（'cpu'或'cuda'）
    patience : int
        早停耐心值
    
    Returns:
    --------
    Dict : 训练历史
    """
    model = model.to(device)
    
    # 损失函数
    criterion_menstruation = nn.CrossEntropyLoss()
    criterion_pain = nn.MSELoss()
    
    # 优化器 - 使用更强的权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))

    # 学习率调度器：预热 + 余弦退火
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 预热阶段：线性增加到初始学习率
            return (epoch + 1) / warmup_epochs
        else:
            # 余弦退火阶段
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_menstruation_acc': [], 'val_menstruation_acc': [],
        'train_pain_mae': [], 'val_pain_mae': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_start_time = time.time()
    logger.info("=" * 80)
    logger.info("开始训练模型")
    logger.info("=" * 80)
    logger.info(f"设备: {device}")
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    logger.info(f"总训练轮数: {num_epochs}, 早停耐心值: {patience}")
    logger.info(f"梯度累积步数: {gradient_accumulation_steps}")
    logger.info(f"学习率预热轮数: {warmup_epochs}")
    print("\n" + "=" * 80)
    print("开始训练模型")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
    print(f"总训练轮数: {num_epochs}, 早停耐心值: {patience}")
    print(f"梯度累积步数: {gradient_accumulation_steps}")
    print(f"学习率预热轮数: {warmup_epochs}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch+1}/{num_epochs} 开始")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_menstruation_correct = 0
        train_menstruation_total = 0
        train_pain_errors = []

        # 使用tqdm显示训练进度（如果可用）
        if HAS_TQDM:
            train_iterator = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}", unit="批次",
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            train_iterator = train_loader

        # 梯度累积训练
        accumulation_counter = 0
        optimizer.zero_grad()  # 在循环外初始化梯度

        for X_batch, y_menstruation_batch, y_pain_batch in train_iterator:
            X_batch = X_batch.to(device)
            y_menstruation_batch = y_menstruation_batch.to(device)
            y_pain_batch = y_pain_batch.to(device)
            
            # 前向传播
            menstruation_logits, pain_pred = model(X_batch)

            # 计算损失（多任务损失加权）
            loss_menstruation = criterion_menstruation(menstruation_logits, y_menstruation_batch)
            loss_pain = criterion_pain(pain_pred, y_pain_batch)
            loss = (loss_menstruation + 0.5 * loss_pain) / gradient_accumulation_steps  # 归一化损失

            # 反向传播（累积梯度）
            loss.backward()

            accumulation_counter += 1

            # 达到累积步数时更新参数
            if accumulation_counter % gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # 统计
            train_loss += loss.item()
            train_menstruation_correct += (menstruation_logits.argmax(1) == y_menstruation_batch).sum().item()
            train_menstruation_total += y_menstruation_batch.size(0)
            train_pain_errors.extend((pain_pred - y_pain_batch).abs().cpu().numpy())
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_menstruation_correct = 0
        val_menstruation_total = 0
        val_pain_errors = []
        
        with torch.no_grad():
            for X_batch, y_menstruation_batch, y_pain_batch in val_loader:
                X_batch = X_batch.to(device)
                y_menstruation_batch = y_menstruation_batch.to(device)
                y_pain_batch = y_pain_batch.to(device)
                
                menstruation_logits, pain_pred = model(X_batch)
                
                loss_menstruation = criterion_menstruation(menstruation_logits, y_menstruation_batch)
                loss_pain = criterion_pain(pain_pred, y_pain_batch)
                loss = loss_menstruation + 0.5 * loss_pain
                
                val_loss += loss.item()
                val_menstruation_correct += (menstruation_logits.argmax(1) == y_menstruation_batch).sum().item()
                val_menstruation_total += y_menstruation_batch.size(0)
                val_pain_errors.extend((pain_pred - y_pain_batch).abs().cpu().numpy())
        
        # 计算平均指标
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_menstruation_correct / train_menstruation_total
        val_acc = val_menstruation_correct / val_menstruation_total
        train_mae = np.mean(train_pain_errors)
        val_mae = np.mean(val_pain_errors)
        
        # 更新学习率 (LambdaLR 需要传入当前epoch)
        scheduler.step(epoch)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_menstruation_acc'].append(train_acc)
        history['val_menstruation_acc'].append(val_acc)
        history['train_pain_mae'].append(train_mae)
        history['val_pain_mae'].append(val_mae)
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log_msg = (f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"训练 Loss: {train_loss:.4f}, 月经准确率: {train_acc:.4f}, 疼痛MAE: {train_mae:.4f} | "
                      f"验证 Loss: {val_loss:.4f}, 月经准确率: {val_acc:.4f}, 疼痛MAE: {val_mae:.4f} | "
                      f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
            logger.info(log_msg)
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  训练 - Loss: {train_loss:.4f}, 月经准确率: {train_acc:.4f}, 疼痛MAE: {train_mae:.4f}")
            print(f"  验证 - Loss: {val_loss:.4f}, 月经准确率: {val_acc:.4f}, 疼痛MAE: {val_mae:.4f}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f"Epoch [{epoch+1}] - 发现更好的模型，验证损失: {val_loss:.4f}，已保存")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.warning(f"早停触发！在第 {epoch+1} 轮停止训练（验证损失连续 {patience} 轮未改善）")
                print(f"\n早停触发！在第 {epoch+1} 轮停止训练")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    logger.info(f"训练完成！最佳验证损失: {best_val_loss:.4f}")
    print("\n训练完成！")
    
    return history


def evaluate_personalized_model(model, test_loader, device='cpu'):
    """
    评估个性化模型

    Parameters:
    -----------
    model : nn.Module
        训练好的个性化模型
    test_loader : DataLoader
        测试数据加载器（包含用户特征）
    device : str
        设备

    Returns:
    --------
    Dict : 评估指标
    """
    logger.info("开始评估个性化模型...")
    model = model.to(device)
    model.eval()

    all_menstruation_pred = []
    all_menstruation_true = []
    all_menstruation_proba = []
    all_pain_pred = []
    all_pain_true = []

    with torch.no_grad():
        for X_batch, X_user_batch, y_menstruation_batch, y_pain_batch in test_loader:
            X_batch = X_batch.to(device)
            X_user_batch = X_user_batch.to(device)

            menstruation_logits, pain_pred = model(X_batch, X_user_batch)

            all_menstruation_pred.extend(menstruation_logits.argmax(1).cpu().numpy())
            all_menstruation_true.extend(y_menstruation_batch.numpy())
            all_menstruation_proba.extend(torch.softmax(menstruation_logits, dim=1)[:, 1].cpu().numpy())
            all_pain_pred.extend(pain_pred.cpu().numpy())
            all_pain_true.extend(y_pain_batch.numpy())

    # 月经预测指标（分类）
    menstruation_acc = accuracy_score(all_menstruation_true, all_menstruation_pred)
    menstruation_precision = precision_score(all_menstruation_true, all_menstruation_pred)
    menstruation_recall = recall_score(all_menstruation_true, all_menstruation_pred)
    menstruation_f1 = f1_score(all_menstruation_true, all_menstruation_pred)
    menstruation_auc = roc_auc_score(all_menstruation_true, all_menstruation_proba)

    # 疼痛预测指标（回归）
    pain_mae = mean_absolute_error(all_pain_true, all_pain_pred)
    pain_rmse = np.sqrt(mean_squared_error(all_pain_true, all_pain_pred))
    pain_r2 = r2_score(all_pain_true, all_pain_pred)

    # 仅计算月经期的疼痛预测指标
    menstruation_mask = np.array(all_menstruation_true) == 1
    if menstruation_mask.sum() > 0:
        pain_mae_menstruation = mean_absolute_error(
            np.array(all_pain_true)[menstruation_mask],
            np.array(all_pain_pred)[menstruation_mask]
        )
    else:
        pain_mae_menstruation = 0.0

    metrics = {
        'menstruation': {
            'accuracy': menstruation_acc,
            'precision': menstruation_precision,
            'recall': menstruation_recall,
            'f1_score': menstruation_f1,
            'auc_roc': menstruation_auc
        },
        'pain': {
            'mae': pain_mae,
            'rmse': pain_rmse,
            'r2_score': pain_r2,
            'mae_menstruation_only': pain_mae_menstruation
        }
    }

    logger.info("个性化模型评估完成")
    logger.info(f"月经预测 - 准确率: {menstruation_acc:.4f}, F1: {menstruation_f1:.4f}, AUC: {menstruation_auc:.4f}")
    logger.info(f"疼痛预测 - MAE: {pain_mae:.4f}, RMSE: {pain_rmse:.4f}, R²: {pain_r2:.4f}")

    return metrics


def evaluate_model(model, test_loader, device='cpu'):
    """
    评估模型（兼容旧版本）

    Parameters:
    -----------
    model : nn.Module
        训练好的模型
    test_loader : DataLoader
        测试数据加载器
    device : str
        设备

    Returns:
    --------
    Dict : 评估指标
    """
    logger.info("开始评估模型...")
    model = model.to(device)
    model.eval()

    all_menstruation_pred = []
    all_menstruation_true = []
    all_menstruation_proba = []
    all_pain_pred = []
    all_pain_true = []

    with torch.no_grad():
        for X_batch, y_menstruation_batch, y_pain_batch in test_loader:
            X_batch = X_batch.to(device)

            menstruation_logits, pain_pred = model(X_batch)

            all_menstruation_pred.extend(menstruation_logits.argmax(1).cpu().numpy())
            all_menstruation_true.extend(y_menstruation_batch.numpy())
            all_menstruation_proba.extend(torch.softmax(menstruation_logits, dim=1)[:, 1].cpu().numpy())
            all_pain_pred.extend(pain_pred.cpu().numpy())
            all_pain_true.extend(y_pain_batch.numpy())

    # 月经预测指标（分类）
    menstruation_acc = accuracy_score(all_menstruation_true, all_menstruation_pred)
    menstruation_precision = precision_score(all_menstruation_true, all_menstruation_pred)
    menstruation_recall = recall_score(all_menstruation_true, all_menstruation_pred)
    menstruation_f1 = f1_score(all_menstruation_true, all_menstruation_pred)
    menstruation_auc = roc_auc_score(all_menstruation_true, all_menstruation_proba)

    # 疼痛预测指标（回归）
    pain_mae = mean_absolute_error(all_pain_true, all_pain_pred)
    pain_rmse = np.sqrt(mean_squared_error(all_pain_true, all_pain_pred))
    pain_r2 = r2_score(all_pain_true, all_pain_pred)

    # 仅计算月经期的疼痛预测指标
    menstruation_mask = np.array(all_menstruation_true) == 1
    if menstruation_mask.sum() > 0:
        pain_mae_menstruation = mean_absolute_error(
            np.array(all_pain_true)[menstruation_mask],
            np.array(all_pain_pred)[menstruation_mask]
        )
    else:
        pain_mae_menstruation = 0.0

    metrics = {
        'menstruation': {
            'accuracy': menstruation_acc,
            'precision': menstruation_precision,
            'recall': menstruation_recall,
            'f1_score': menstruation_f1,
            'auc_roc': menstruation_auc
        },
        'pain': {
            'mae': pain_mae,
            'rmse': pain_rmse,
            'r2_score': pain_r2,
            'mae_menstruation_only': pain_mae_menstruation
        }
    }

    logger.info("模型评估完成")
    logger.info(f"月经预测 - 准确率: {menstruation_acc:.4f}, F1: {menstruation_f1:.4f}, AUC: {menstruation_auc:.4f}")
    logger.info(f"疼痛预测 - MAE: {pain_mae:.4f}, RMSE: {pain_rmse:.4f}, R²: {pain_r2:.4f}")

    return metrics


def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='训练损失')
    axes[0, 0].plot(history['val_loss'], label='验证损失')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 月经准确率
    axes[0, 1].plot(history['train_menstruation_acc'], label='训练准确率')
    axes[0, 1].plot(history['val_menstruation_acc'], label='验证准确率')
    axes[0, 1].set_title('月经预测准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 疼痛MAE
    axes[1, 0].plot(history['train_pain_mae'], label='训练MAE')
    axes[1, 0].plot(history['val_pain_mae'], label='验证MAE')
    axes[1, 0].set_title('疼痛预测MAE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"训练历史图已保存到: {save_path}")
    print(f"训练历史图已保存到: {save_path}")
    plt.close()


def train_personalized_lstm():
    """训练个性化LSTM模型的主函数"""
    logger.info("=" * 80)
    logger.info("个性化LSTM模型 - 女性健康管理时间序列预测")
    logger.info("=" * 80)
    logger.info(f"程序启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("🎯 个性化LSTM模型 - 女性健康管理时间序列预测")
    print("基于用户个性化特征的精准预测系统")
    print("=" * 80)
    print()

    # 显示硬件信息
    print("硬件配置:")
    print("  CPU: Intel Core i9-14995HX (高性能移动处理器)")
    print("  内存: 512GB (大幅提升，充分利用大内存优势)")
    print("  存储: 1TB SSD")
    print("  设备: " + ('CUDA GPU' if torch.cuda.is_available() else 'CPU'))
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # 个性化配置参数 - 针对14995HX/512GB内存/1TB SSD优化
    config = {
        'data_path': '../data_generation/lstm_dataset.csv',
        'user_attrs_path': '../data_generation/lstm_dataset_user_attributes.csv',
        'window_size': 30,  # 时间窗口大小（天）
        'batch_size': 1024,  # 个性化模型batch_size适中（充分利用内存）
        'hidden_size': 256,  # 个性化模型隐藏单元
        'num_layers': 3,  # LSTM层数
        'dropout': 0.25,  # dropout率
        'learning_rate': 0.001,
        'num_epochs': 20,  # 个性化训练轮数
        'patience': 8,  # 早停耐心值
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 12,  # workers数量
        'pin_memory': True if torch.cuda.is_available() else False,
        'gradient_accumulation_steps': 2,  # 梯度累积
        'warmup_epochs': 3,  # 预热轮数
    }

    logger.info("个性化配置参数:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    print("个性化配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # 加载和预处理个性化数据
    data = load_and_preprocess_data_personalized(
        data_path=config['data_path'],
        user_attrs_path=config['user_attrs_path'],
        window_size=config['window_size']
    )

    # 创建个性化数据加载器
    train_dataset = PersonalizedTimeSeriesDataset(
        data['X_train'], data['X_user_train'],
        data['y_menstruation_train'], data['y_pain_train']
    )
    val_dataset = PersonalizedTimeSeriesDataset(
        data['X_val'], data['X_user_val'],
        data['y_menstruation_val'], data['y_pain_val']
    )
    test_dataset = PersonalizedTimeSeriesDataset(
        data['X_test'], data['X_user_test'],
        data['y_menstruation_test'], data['y_pain_test']
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        persistent_workers=True if config.get('num_workers', 0) > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        persistent_workers=True if config.get('num_workers', 0) > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        persistent_workers=True if config.get('num_workers', 0) > 0 else False
    )

    # 创建个性化模型
    model = PersonalizedMultiTaskLSTM(
        input_size=data['input_size'],
        user_feature_size=data['user_feature_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    logger.info(f"个性化模型架构:")
    logger.info(f"  时间序列输入维度: {data['input_size']}")
    logger.info(f"  用户特征维度: {data['user_feature_size']}")
    logger.info(f"  隐藏单元: {config['hidden_size']}")
    logger.info(f"  LSTM层数: {config['num_layers']}")
    logger.info(f"  Dropout: {config['dropout']}")
    logger.info(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n个性化模型架构:")
    print(model)
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练个性化模型
    history = train_personalized_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device'],
        patience=config['patience'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        warmup_epochs=config.get('warmup_epochs', 0)
    )

    # 评估个性化模型
    print("\n" + "=" * 80)
    print("🎯 个性化模型评估")
    print("=" * 80)
    metrics = evaluate_personalized_model(model, test_loader, device=config['device'])

    logger.info("\n个性化月经预测指标（分类）:")
    for key, value in metrics['menstruation'].items():
        logger.info(f"  {key}: {value:.4f}")
    print("\n个性化月经预测指标（分类）:")
    for key, value in metrics['menstruation'].items():
        print(f"  {key}: {value:.4f}")

    logger.info("\n个性化疼痛等级预测指标（回归）:")
    for key, value in metrics['pain'].items():
        logger.info(f"  {key}: {value:.4f}")
    print("\n个性化疼痛等级预测指标（回归）:")
    for key, value in metrics['pain'].items():
        print(f"  {key}: {value:.4f}")

    # 保存个性化评估结果
    with open('personalized_model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("个性化评估结果已保存到: personalized_model_metrics.json")
    print("\n个性化评估结果已保存到: personalized_model_metrics.json")

    # 绘制训练历史
    plot_training_history(history, save_path='personalized_training_history.png')

    # 保存个性化预处理器
    preprocessor_path = 'personalized_preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(data['preprocessor'], f)
    logger.info(f"个性化预处理器已保存到: {preprocessor_path}")
    print(f"个性化预处理器已保存到: {preprocessor_path}")

    # 保存个性化完整模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'input_size': data['input_size'],
        'user_feature_size': data['user_feature_size'],
        'metrics': metrics,
        'preprocessor_path': preprocessor_path
    }, 'personalized_lstm_model_complete.pth')
    logger.info("个性化完整模型已保存到: personalized_lstm_model_complete.pth")
    print("个性化完整模型已保存到: personalized_lstm_model_complete.pth")

    logger.info("=" * 80)
    logger.info("个性化训练完成！")
    logger.info(f"程序结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    print("\n" + "=" * 80)
    print("🎉 个性化训练完成！")
    print("现在可以根据每个用户的个性化特征进行精准预测")
    print("=" * 80)


def main():
    """主函数 - 支持选择训练模式"""
    import argparse

    parser = argparse.ArgumentParser(description='LSTM模型训练')
    parser.add_argument('--personalized', action='store_true',
                       help='使用个性化训练模式')
    parser.add_argument('--legacy', action='store_true',
                       help='使用传统训练模式（兼容旧版本）')

    args = parser.parse_args()

    if args.personalized:
        # 运行个性化训练
        train_personalized_lstm()
    else:
        # 运行传统训练（默认）
        logger.info("=" * 80)
        logger.info("LSTM模型 - 女性健康管理时间序列预测（传统模式）")
        logger.info("=" * 80)
        logger.info(f"程序启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print("LSTM模型 - 女性健康管理时间序列预测（传统模式）")
        print("=" * 80)
        print()

        # 显示硬件信息
        print("硬件配置:")
        print("  CPU: Intel Core i9-14995HX (高性能移动处理器)")
        print("  内存: 512GB (大幅提升，充分利用大内存优势)")
        print("  存储: 1TB SSD")
        print("  设备: " + ('CUDA GPU' if torch.cuda.is_available() else 'CPU'))
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print()

        # 配置参数 - 针对14995HX/512GB内存/1TB SSD优化
        config = {
            'data_path': '../data_generation/lstm_dataset.csv',
            'window_size': 30,  # 时间窗口大小（天）
            'batch_size': 2048,  # 大幅增大batch_size充分利用512GB内存
            'hidden_size': 384,  # 隐藏单元数
            'num_layers': 3,  # LSTM层数
            'dropout': 0.25,  # dropout率
            'learning_rate': 0.001,
            'num_epochs': 30,  # 训练轮数
            'patience': 10,  # 早停耐心值
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 16,  # workers数量
            'pin_memory': True if torch.cuda.is_available() else False,
            'gradient_accumulation_steps': 2,  # 梯度累积
            'warmup_epochs': 3,  # 预热轮数
        }

        logger.info("配置参数:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        print("配置参数:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

        # 加载和预处理数据
        data = load_and_preprocess_data(
            data_path=config['data_path'],
            window_size=config['window_size']
        )

        # 创建数据加载器
        train_dataset = TimeSeriesDataset(
            data['X_train'],
            data['y_menstruation_train'],
            data['y_pain_train']
        )
        val_dataset = TimeSeriesDataset(
            data['X_val'],
            data['y_menstruation_val'],
            data['y_pain_val']
        )
        test_dataset = TimeSeriesDataset(
            data['X_test'],
            data['y_menstruation_test'],
            data['y_pain_test']
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
            persistent_workers=True if config.get('num_workers', 0) > 0 else False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
            persistent_workers=True if config.get('num_workers', 0) > 0 else False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
            persistent_workers=True if config.get('num_workers', 0) > 0 else False
        )

        # 创建模型
        model = MultiTaskLSTM(
            input_size=data['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )

        logger.info(f"模型架构:")
        logger.info(f"  输入维度: {data['input_size']}")
        logger.info(f"  隐藏单元: {config['hidden_size']}")
        logger.info(f"  LSTM层数: {config['num_layers']}")
        logger.info(f"  Dropout: {config['dropout']}")
        logger.info(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"\n模型架构:")
        print(model)
        print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # 训练模型
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            device=config['device'],
            patience=config['patience'],
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            warmup_epochs=config.get('warmup_epochs', 0)
        )

        # 评估模型
        print("\n" + "=" * 80)
        print("模型评估")
        print("=" * 80)
        metrics = evaluate_model(model, test_loader, device=config['device'])

        logger.info("\n月经预测指标（分类）:")
        for key, value in metrics['menstruation'].items():
            logger.info(f"  {key}: {value:.4f}")
        print("\n月经预测指标（分类）:")
        for key, value in metrics['menstruation'].items():
            print(f"  {key}: {value:.4f}")

        logger.info("\n疼痛等级预测指标（回归）:")
        for key, value in metrics['pain'].items():
            logger.info(f"  {key}: {value:.4f}")
        print("\n疼痛等级预测指标（回归）:")
        for key, value in metrics['pain'].items():
            print(f"  {key}: {value:.4f}")

        # 保存评估结果
        with open('model_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("评估结果已保存到: model_metrics.json")
        print("\n评估结果已保存到: model_metrics.json")

        # 绘制训练历史
        plot_training_history(history)

        # 保存预处理器
        preprocessor_path = 'preprocessor.pkl'
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(data['preprocessor'], f)
        logger.info(f"预处理器已保存到: {preprocessor_path}")
        print(f"预处理器已保存到: {preprocessor_path}")

        # 保存完整模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'input_size': data['input_size'],
            'metrics': metrics,
            'preprocessor_path': preprocessor_path
        }, 'lstm_model_complete.pth')
        logger.info("完整模型已保存到: lstm_model_complete.pth")
        print("完整模型已保存到: lstm_model_complete.pth")

        logger.info("=" * 80)
        logger.info("训练完成！")
        logger.info(f"程序结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)


if __name__ == '__main__':
    main()

