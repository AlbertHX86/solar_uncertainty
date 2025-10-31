"""
高斯过程近似模型 - 预测均值和方差
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_encoder import TCNBiLSTMEncoder


class GPApproximationModel(nn.Module):
    """
    高斯过程近似模型
    使用神经网络预测均值和方差，近似高斯过程
    
    Architecture:
        Input -> TCNBiLSTMEncoder -> FC_mean (均值)
                                  -> FC_var (方差)
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
        super(GPApproximationModel, self).__init__()
        
        # 编码器
        self.encoder = TCNBiLSTMEncoder(
            n_features=n_features,
            n_filters=n_filters,
            n_filters1=n_filters1,
            filter_size=filter_size,
            dropout_rate=dropout_rate
        )
        
        # 均值预测头
        self.fc_mean = nn.Linear(self.encoder.output_dim, 1)
        
        # 方差预测头（输出log variance以保证方差为正）
        self.fc_var = nn.Linear(self.encoder.output_dim, 1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, n_features]
            
        Returns:
            mean: [batch_size] - 预测均值
            var: [batch_size] - 预测方差
        """
        # 编码
        encoded = self.encoder.get_last_hidden(x)  # [batch_size, n_filters1*2]
        
        # 预测均值
        mean = self.fc_mean(encoded).squeeze(-1)  # [batch_size]
        
        # 预测方差（使用softplus保证为正）
        log_var = self.fc_var(encoded).squeeze(-1)  # [batch_size]
        var = F.softplus(log_var) + 1e-6  # 添加小常数避免除零
        
        return mean, var
    
    def predict_with_uncertainty(self, x, confidence=0.8):
        """
        预测并返回不确定性区间
        
        Args:
            x: [batch_size, seq_len, n_features]
            confidence: 置信水平（默认80%）
            
        Returns:
            lower: 下界
            median: 中位数（均值）
            upper: 上界
        """
        mean, var = self.forward(x)
        std = torch.sqrt(var)
        
        # 根据置信水平计算z值
        # 80%置信区间对应z=1.28，90%对应z=1.645
        if confidence == 0.8:
            z = 1.28
        elif confidence == 0.9:
            z = 1.645
        else:
            z = 1.28
        
        lower = mean - z * std
        upper = mean + z * std
        median = mean
        
        return lower, median, upper


def gaussian_nll_loss(mean, var, target):
    """
    高斯负对数似然损失
    
    Args:
        mean: [batch_size] - 预测均值
        var: [batch_size] - 预测方差
        target: [batch_size] - 真实值
        
    Returns:
        loss: 标量损失值
    """
    # 负对数似然: 0.5 * (log(var) + (target - mean)^2 / var)
    loss = 0.5 * (torch.log(var) + (target - mean) ** 2 / var)
    
    return loss.mean()

