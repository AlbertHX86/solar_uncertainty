"""
统一的TCN-BiLSTM编码器
所有模型共享相同的特征提取器，保证公平对比
"""
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from TCN import TCNBlock


class TCNBiLSTMEncoder(nn.Module):
    """
    统一的TCN-BiLSTM编码器
    
    Architecture:
        Input -> TCN -> Dropout -> BiLSTM -> Encoded Features
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
        super(TCNBiLSTMEncoder, self).__init__()
        
        # TCN层
        self.tcn = TCNBlock(
            n_features=n_features,
            n_filters=n_filters,
            filter_size=filter_size
        )
        
        # Dropout层
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=n_filters,
            hidden_size=n_filters1,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.n_features = n_features
        self.n_filters = n_filters
        self.n_filters1 = n_filters1
        self.output_dim = n_filters1 * 2  # BiLSTM输出是双向的
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, n_features]
            
        Returns:
            encoded: [batch_size, seq_len, n_filters1*2]
        """
        # TCN特征提取
        tcn_out = self.tcn(x)  # [batch_size, seq_len, n_filters]
        
        # Dropout
        tcn_out = self.dropout(tcn_out)
        
        # BiLSTM编码
        lstm_out, (h_n, c_n) = self.bilstm(tcn_out)  # [batch_size, seq_len, n_filters1*2]
        
        return lstm_out
    
    def get_last_hidden(self, x):
        """
        获取最后一个时间步的隐藏状态
        
        Args:
            x: [batch_size, seq_len, n_features]
            
        Returns:
            last_hidden: [batch_size, n_filters1*2]
        """
        lstm_out = self.forward(x)
        return lstm_out[:, -1, :]  # 取最后一个时间步

