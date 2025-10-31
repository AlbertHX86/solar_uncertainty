"""
分位数回归模型 - 直接预测不同分位数
"""
import torch
import torch.nn as nn
from .base_encoder import TCNBiLSTMEncoder


class QuantileRegressionModel(nn.Module):
    """
    分位数回归模型
    直接预测三个分位数：下界(10%)、中位数(50%)、上界(90%)
    
    Architecture:
        Input -> TCNBiLSTMEncoder -> FC -> [Q10, Q50, Q90]
    """
    
    def __init__(self, n_features, n_filters, n_filters1, filter_size, 
                 quantiles=[0.1, 0.5, 0.9], dropout_rate=0.1):
        """
        Args:
            n_features: 输入特征维度
            n_filters: TCN卷积核数量
            n_filters1: BiLSTM隐藏层大小
            filter_size: TCN卷积核大小
            quantiles: 预测的分位数列表
            dropout_rate: Dropout比率
        """
        super(QuantileRegressionModel, self).__init__()
        
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        
        # 编码器
        self.encoder = TCNBiLSTMEncoder(
            n_features=n_features,
            n_filters=n_filters,
            n_filters1=n_filters1,
            filter_size=filter_size,
            dropout_rate=dropout_rate
        )
        
        # 输出层（输出多个分位数）
        self.fc = nn.Linear(self.encoder.output_dim, self.n_quantiles)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, n_features]
            
        Returns:
            quantiles: [batch_size, n_quantiles] - 各分位数预测值
        """
        # 编码
        encoded = self.encoder.get_last_hidden(x)  # [batch_size, n_filters1*2]
        
        # 预测分位数
        quantile_preds = self.fc(encoded)  # [batch_size, n_quantiles]
        
        return quantile_preds
    
    def predict_with_uncertainty(self, x):
        """
        预测并返回不确定性区间
        
        Args:
            x: [batch_size, seq_len, n_features]
            
        Returns:
            lower: 下界 (10%)
            median: 中位数 (50%)
            upper: 上界 (90%)
        """
        quantile_preds = self.forward(x)  # [batch_size, 3]
        
        lower = quantile_preds[:, 0]   # Q10
        median = quantile_preds[:, 1]  # Q50
        upper = quantile_preds[:, 2]   # Q90
        
        return lower, median, upper


def quantile_loss(predictions, targets, quantiles=[0.1, 0.5, 0.9]):
    """
    分位数损失函数
    
    Args:
        predictions: [batch_size, n_quantiles]
        targets: [batch_size]
        quantiles: 分位数列表
        
    Returns:
        loss: 标量损失值
    """
    targets = targets.unsqueeze(1)  # [batch_size, 1]
    errors = targets - predictions  # [batch_size, n_quantiles]
    
    quantiles_tensor = torch.tensor(quantiles, device=predictions.device).view(1, -1)
    
    # 分位数损失
    loss = torch.max(quantiles_tensor * errors, (quantiles_tensor - 1) * errors)
    
    return loss.mean()

