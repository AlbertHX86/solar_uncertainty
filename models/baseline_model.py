"""
基准点预测模型 - 不提供不确定性估计
"""
import torch
import torch.nn as nn
from .base_encoder import TCNBiLSTMEncoder


class BaselinePointModel(nn.Module):
    """
    基准点预测模型
    仅提供点预测，不提供不确定性估计
    
    Architecture:
        Input -> TCNBiLSTMEncoder -> FC -> Point Prediction
    """
    
    def __init__(self, n_features, n_filters, n_filters1, filter_size, dropout_rate=0.1):
        """
        Args:
            n_features: 输入特征维度
            n_filters: TCN卷积核数量
            n_filters1: BiLSTM隐藏层大小
            filter_size: TCN卷积核大小
            dropout_rate: Dropout比率
        """
        super(BaselinePointModel, self).__init__()
        
        # 编码器
        self.encoder = TCNBiLSTMEncoder(
            n_features=n_features,
            n_filters=n_filters,
            n_filters1=n_filters1,
            filter_size=filter_size,
            dropout_rate=dropout_rate
        )
        
        # 输出层
        self.fc = nn.Linear(self.encoder.output_dim, 1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, n_features]
            
        Returns:
            prediction: [batch_size] - 点预测值
        """
        # 编码
        encoded = self.encoder.get_last_hidden(x)  # [batch_size, n_filters1*2]
        
        # 预测
        prediction = self.fc(encoded).squeeze(-1)  # [batch_size]
        
        return prediction

