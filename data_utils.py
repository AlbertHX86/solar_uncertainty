"""
数据加载和预处理工具
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(config):
    """
    加载并预处理多特征数据
    
    Args:
        config: 配置对象
        
    Returns:
        data_train: 训练集
        data_test: 测试集
        scaler: 归一化器
        feature_names: 特征名称列表
        data_dim: 特征维度
    """
    print(f"\n[数据加载] 从 {config.data_path} 加载数据...")
    
    # 读取Excel文件
    df = pd.read_excel(config.data_path)
    
    # 特征列（排除date和power）
    feature_columns = [
        'Total solar irradiance (W/m2)',
        'Direct normal irradiance (W/m2)',
        'Global horizontal irradiance (W/m2)',
        'Air temperature  (°C) ',
        'Atmosphere (hpa)',
        'Relative humidity (%)'
    ]
    
    # 目标列
    target_column = 'power'
    
    # 处理数据类型（某些列可能是object类型）
    for col in feature_columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除缺失值
    df_clean = df.dropna()
    print(f"  原始数据: {len(df)} 行")
    print(f"  清洗后: {len(df_clean)} 行")
    
    # 提取特征和目标
    features = df_clean[feature_columns].values
    target = df_clean[target_column].values.reshape(-1, 1)
    
    # 合并特征和目标
    data = np.concatenate([features, target], axis=1)
    data_dim = data.shape[1]
    
    print(f"\n[特征信息]")
    print(f"  特征维度: {data_dim}")
    print(f"  特征列表:")
    for i, name in enumerate(feature_columns + [target_column]):
        print(f"    {i+1}. {name}")
    
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # 划分训练集和测试集
    train_size = int(len(data_scaled) * config.train_ratio)
    data_train = data_scaled[:train_size]
    data_test = data_scaled[train_size:]
    
    print(f"\n[数据划分]")
    print(f"  训练集: {len(data_train)} 样本 ({config.train_ratio*100:.0f}%)")
    print(f"  测试集: {len(data_test)} 样本 ({(1-config.train_ratio)*100:.0f}%)")
    
    return data_train, data_test, scaler, feature_columns + [target_column], data_dim


def create_sequences(data, window, predict_length):
    """
    创建时间序列窗口
    
    Args:
        data: 输入数据
        window: 时间窗口长度
        predict_length: 预测步长
        
    Returns:
        X: 输入序列
        y: 目标值
    """
    sequence_length = window + predict_length
    sequences = []
    
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    
    sequences = np.array(sequences)
    X = sequences[:, :window, :]  # 输入窗口
    y = sequences[:, -predict_length:, -1]  # 目标值（最后一列）
    
    return X, y


def create_dataloader(data, window, predict_length, batch_size, data_dim, shuffle=True, pin_memory=True):
    """
    创建PyTorch DataLoader
    
    Args:
        data: 输入数据
        window: 时间窗口长度
        predict_length: 预测步长
        batch_size: 批次大小
        data_dim: 特征维度
        shuffle: 是否打乱数据
        pin_memory: 是否使用pin_memory（GPU加速）
        
    Returns:
        dataloader: DataLoader对象
        X_tensor: 输入张量
        y_tensor: 目标张量
    """
    # 创建序列
    X, y = create_sequences(data, window, predict_length)
    
    # 转换为张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # 创建数据集
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # 创建DataLoader（GPU优化）
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory and torch.cuda.is_available(),
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    return dataloader, X_tensor, y_tensor


def inverse_transform_predictions(predictions, scaler, feature_dim):
    """
    反归一化预测结果
    
    Args:
        predictions: 归一化的预测值
        scaler: MinMaxScaler对象
        feature_dim: 特征维度
        
    Returns:
        original_scale: 原始尺度的预测值
    """
    # 创建完整的特征数组（其他特征填0）
    dummy_features = np.zeros((len(predictions), feature_dim))
    dummy_features[:, -1] = predictions  # 最后一列是power
    
    # 反归一化
    original_scale = scaler.inverse_transform(dummy_features)[:, -1]
    
    # 光伏发电不能为负
    original_scale = np.maximum(original_scale, 0)
    
    return original_scale

