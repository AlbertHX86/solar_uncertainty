"""
NPKDE (Non-Parametric Kernel Density Estimation) - 传统统计方法
"""
import numpy as np
from sklearn.neighbors import KernelDensity


class NPKDEUncertainty:
    """
    非参数核密度估计不确定性量化
    
    基于点预测模型的残差分布，使用核密度估计建模不确定性
    这是一个传统的统计方法，作为对比基准
    """
    
    def __init__(self, point_predictions, true_values, bandwidth='scott'):
        """
        Args:
            point_predictions: 点预测结果（训练集）
            true_values: 真实值（训练集）
            bandwidth: KDE带宽选择方法
        """
        # 计算残差
        residuals = true_values - point_predictions
        self.residuals = residuals.reshape(-1, 1)
        
        # 拟合核密度估计
        self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        self.kde.fit(self.residuals)
        
        print(f"  NPKDE初始化完成")
        print(f"    残差均值: {residuals.mean():.4f}")
        print(f"    残差标准差: {residuals.std():.4f}")
        print(f"    残差范围: [{residuals.min():.4f}, {residuals.max():.4f}]")
    
    def predict_intervals(self, point_predictions, n_samples=1000, confidence=0.8):
        """
        基于点预测和残差分布，生成预测区间
        
        Args:
            point_predictions: 点预测结果（测试集）
            n_samples: 采样数量
            confidence: 置信水平
            
        Returns:
            lower: 下界
            median: 中位数
            upper: 上界
        """
        n_predictions = len(point_predictions)
        
        # 从核密度估计中采样残差
        sampled_residuals = self.kde.sample(n_samples)  # [n_samples, 1]
        sampled_residuals = sampled_residuals.flatten()
        
        # 为每个预测点生成不确定性区间
        lower = np.zeros(n_predictions)
        median = np.zeros(n_predictions)
        upper = np.zeros(n_predictions)
        
        # 计算分位数
        alpha = (1 - confidence) / 2
        q_lower = alpha
        q_upper = 1 - alpha
        
        for i in range(n_predictions):
            # 点预测 + 采样残差 = 预测分布
            predicted_distribution = point_predictions[i] + sampled_residuals
            
            # 计算分位数
            lower[i] = np.quantile(predicted_distribution, q_lower)
            median[i] = np.quantile(predicted_distribution, 0.5)
            upper[i] = np.quantile(predicted_distribution, q_upper)
        
        return lower, median, upper
    
    def estimate_uncertainty_scores(self, point_predictions):
        """
        估计每个预测点的不确定性分数
        
        Args:
            point_predictions: 点预测结果
            
        Returns:
            uncertainty_scores: 不确定性分数（方差估计）
        """
        n_predictions = len(point_predictions)
        n_samples = 1000
        
        # 从核密度估计中采样
        sampled_residuals = self.kde.sample(n_samples).flatten()
        
        # 估计每个点的不确定性（残差分布的标准差）
        uncertainty_scores = np.ones(n_predictions) * np.std(sampled_residuals)
        
        return uncertainty_scores

