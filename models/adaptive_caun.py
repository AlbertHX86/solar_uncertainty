"""
Adaptive CAUN - 我们的创新方法
Cross-Attention Uncertainty Network with Adaptive Intervals
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyDetector:
    """
    不确定性检测器
    识别预测性能较差的区域（高不确定性区域）
    """
    
    def __init__(self, threshold_percentile=70):
        """
        Args:
            threshold_percentile: 不确定性阈值百分位数
        """
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    def detect(self, y_true, y_pred):
        """
        检测高不确定性区域
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            high_unc_mask: 高不确定性区域的mask
            unc_scores: 不确定性分数
        """
        # 1. 残差（预测误差）
        residuals = np.abs(y_true - y_pred)
        
        # 2. 局部波动性（使用滑动窗口）
        window = 5
        local_volatility = np.zeros_like(residuals)
        for i in range(len(residuals)):
            start = max(0, i - window)
            end = min(len(residuals), i + window + 1)
            local_volatility[i] = np.std(y_true[start:end])
        
        # 3. 预测变化率（预测的不稳定性）
        pred_change = np.zeros_like(residuals)
        pred_change[1:] = np.abs(np.diff(y_pred))
        
        # 综合不确定性分数（归一化后加权）
        residuals_norm = (residuals - residuals.min()) / (residuals.max() - residuals.min() + 1e-8)
        volatility_norm = (local_volatility - local_volatility.min()) / (local_volatility.max() - local_volatility.min() + 1e-8)
        change_norm = (pred_change - pred_change.min()) / (pred_change.max() - pred_change.min() + 1e-8)
        
        unc_scores = 0.5 * residuals_norm + 0.3 * volatility_norm + 0.2 * change_norm
        
        # 确定阈值
        self.threshold = np.percentile(unc_scores, self.threshold_percentile)
        high_unc_mask = unc_scores > self.threshold
        
        print(f"\n[不确定性检测]")
        print(f"  总样本数: {len(unc_scores)}")
        print(f"  高不确定性样本: {high_unc_mask.sum()} ({high_unc_mask.sum()/len(unc_scores)*100:.1f}%)")
        print(f"  不确定性阈值 (P{self.threshold_percentile}): {self.threshold:.4f}")
        print(f"  不确定性分数范围: [{unc_scores.min():.4f}, {unc_scores.max():.4f}]")
        
        return high_unc_mask, unc_scores


class AdaptiveCAUN(nn.Module):
    """
    Adaptive Cross-Attention Uncertainty Network
    
    创新点：
    1. 使用Transformer Decoder的Cross-Attention机制进行不确定性建模
    2. 基于不确定性分数自适应调整预测区间宽度
    3. 两阶段训练：先识别高不确定性区域，再针对性建模
    
    Architecture:
        Encoded Features (from base encoder) 
            -> Transformer Decoder (Cross-Attention)
            -> Quantile Predictions [Q10, Q50, Q90]
            -> Uncertainty Weight Network
            -> Adaptive Interval Adjustment
    """
    
    def __init__(self, base_encoder, decoder_dim=100, num_heads=5, 
                 num_decoder_layers=2, dropout_rate=0.1):
        """
        Args:
            base_encoder: 预训练的编码器（TCNBiLSTM）
            decoder_dim: Decoder隐藏层维度
            num_heads: 多头注意力的头数
            num_decoder_layers: Decoder层数
            dropout_rate: Dropout比率
        """
        super(AdaptiveCAUN, self).__init__()
        
        self.base_encoder = base_encoder
        self.encoder_dim = base_encoder.encoder.output_dim
        self.decoder_dim = decoder_dim
        
        # 投影层：将编码器输出映射到Decoder维度
        self.encoder_proj = nn.Linear(self.encoder_dim, decoder_dim)
        
        # 可学习的查询向量（用于Decoder）
        self.query_embed = nn.Parameter(torch.randn(1, 1, decoder_dim))
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # 分位数预测头
        self.quantile_head = nn.Linear(decoder_dim, 3)  # [Q10, Q50, Q90]
        
        # 不确定性权重网络（用于自适应调整）
        self.uncertainty_weight_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出0-1之间的权重
        )
    
    def forward(self, x, uncertainty_scores=None):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, n_features]
            uncertainty_scores: [batch_size] - 不确定性分数（可选）
            
        Returns:
            quantile_preds: [batch_size, 3] - 自适应调整后的分位数预测
        """
        batch_size = x.size(0)
        
        # 1. 编码（使用预训练的编码器）
        with torch.no_grad():
            encoded = self.base_encoder.encoder(x)  # [batch_size, seq_len, encoder_dim]
        
        # 2. 投影到Decoder维度
        memory = self.encoder_proj(encoded)  # [batch_size, seq_len, decoder_dim]
        
        # 3. 准备查询向量
        query = self.query_embed.expand(batch_size, -1, -1)  # [batch_size, 1, decoder_dim]
        
        # 4. Transformer Decoder（Cross-Attention）
        decoder_output = self.transformer_decoder(query, memory)  # [batch_size, 1, decoder_dim]
        decoder_output = decoder_output.squeeze(1)  # [batch_size, decoder_dim]
        
        # 5. 预测基础分位数
        base_quantiles = self.quantile_head(decoder_output)  # [batch_size, 3]
        
        # 6. 自适应调整（如果提供了不确定性分数）
        if uncertainty_scores is not None:
            if not isinstance(uncertainty_scores, torch.Tensor):
                uncertainty_scores = torch.tensor(
                    uncertainty_scores, 
                    dtype=torch.float32, 
                    device=x.device
                )
            
            # 计算调整权重
            unc_input = uncertainty_scores.view(-1, 1)  # [batch_size, 1]
            adjust_weight = self.uncertainty_weight_net(unc_input)  # [batch_size, 1]
            
            # 自适应调整区间宽度
            q10, q50, q90 = base_quantiles[:, 0], base_quantiles[:, 1], base_quantiles[:, 2]
            
            # 计算基础区间宽度
            lower_width = q50 - q10
            upper_width = q90 - q50
            
            # 根据不确定性分数调整（高不确定性 -> 更宽的区间）
            adjusted_lower_width = lower_width * (1 + adjust_weight.squeeze())
            adjusted_upper_width = upper_width * (1 + adjust_weight.squeeze())
            
            # 重新组合
            adjusted_quantiles = torch.stack([
                q50 - adjusted_lower_width,
                q50,
                q50 + adjusted_upper_width
            ], dim=1)
            
            return adjusted_quantiles
        else:
            return base_quantiles
    
    def predict_with_uncertainty(self, x, uncertainty_scores=None):
        """
        预测并返回不确定性区间
        
        Args:
            x: [batch_size, seq_len, n_features]
            uncertainty_scores: [batch_size] - 不确定性分数（可选）
            
        Returns:
            lower: 下界 (10%)
            median: 中位数 (50%)
            upper: 上界 (90%)
        """
        quantile_preds = self.forward(x, uncertainty_scores)
        
        lower = quantile_preds[:, 0]
        median = quantile_preds[:, 1]
        upper = quantile_preds[:, 2]
        
        return lower, median, upper


def adaptive_loss(predictions, targets, uncertainty_scores, alpha=0.1, beta=0.05):
    """
    自适应损失函数
    
    结合：
    1. 分位数损失
    2. 区间合理性约束（下界 < 中位数 < 上界）
    3. 区间宽度惩罚（避免过宽的区间）
    4. 不确定性自适应（高不确定性区域允许更宽的区间）
    
    Args:
        predictions: [batch_size, 3] - [Q10, Q50, Q90]
        targets: [batch_size] - 真实值
        uncertainty_scores: [batch_size] - 不确定性分数
        alpha: 区间合理性约束权重
        beta: 区间宽度惩罚权重
        
    Returns:
        loss: 标量损失值
    """
    if not isinstance(uncertainty_scores, torch.Tensor):
        uncertainty_scores = torch.tensor(
            uncertainty_scores, 
            dtype=torch.float32, 
            device=predictions.device
        )
    
    q10, q50, q90 = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    
    # 1. 分位数损失
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=predictions.device).view(1, -1)
    targets_expanded = targets.unsqueeze(1).expand(-1, 3)
    errors = targets_expanded - predictions
    quantile_loss = torch.max(quantiles * errors, (quantiles - 1) * errors).mean()
    
    # 2. 区间合理性约束
    rationality_loss = F.relu(q10 - q50).mean() + F.relu(q50 - q90).mean()
    
    # 3. 区间宽度惩罚（自适应）
    interval_width = (q90 - q10)
    # 低不确定性区域：更强的宽度惩罚
    # 高不确定性区域：较弱的宽度惩罚
    adaptive_penalty = interval_width * (1 - uncertainty_scores)
    width_loss = adaptive_penalty.mean()
    
    # 总损失
    total_loss = quantile_loss + alpha * rationality_loss + beta * width_loss
    
    return total_loss

