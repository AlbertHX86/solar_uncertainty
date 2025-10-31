"""
配置文件 - 统一管理所有超参数
"""
import os

# OpenMP设置（避免多线程冲突）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import torch


class Config:
    """全局配置类"""
    
    # ==================== 设备配置 ====================
    def __init__(self):
        self.setup_device()
    
    def setup_device(self):
        """设置计算设备"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("✓ 使用Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            print("⚠️  使用CPU (建议使用GPU以加速训练)")
    
    # ==================== 数据配置 ====================
    data_path = 'power.xlsx'
    train_ratio = 0.7
    window = 40  # 时间窗口长度
    predict_length = 1  # 预测步长
    
    # ==================== 模型配置 ====================
    # TCN-BiLSTM编码器
    n_filters = 32  # TCN卷积核数量
    n_filters1 = 50  # BiLSTM隐藏层大小
    filter_size = 4  # TCN卷积核大小
    dropout_rate = 0.1
    
    # CAUN特定参数
    decoder_dim = 100
    num_heads = 5
    num_decoder_layers = 2
    uncertainty_threshold_percentile = 70
    
    # ==================== 训练配置 ====================
    batch_size = 64  # GPU可以使用更大的batch size
    learning_rate = 0.001
    
    # 训练轮数
    epochs_baseline = 30
    epochs_quantile = 30
    epochs_gp = 30
    epochs_caun_stage1 = 30
    epochs_caun_stage2 = 30
    
    # ==================== 评估配置 ====================
    target_coverage = 0.8  # 目标覆盖率
    quantiles = [0.1, 0.5, 0.9]  # 预测分位数
    
    # ==================== 保存路径 ====================
    checkpoint_dir = 'checkpoint'
    results_dir = 'results'
    
    # ==================== 可视化配置 ====================
    figsize = (15, 5)
    dpi = 300
    
    @staticmethod
    def clear_gpu_cache():
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def print_gpu_stats(self):
        """打印GPU使用统计"""
        if torch.cuda.is_available():
            print(f"  GPU内存使用: {torch.cuda.memory_allocated(self.device)/1024**2:.1f}MB / "
                  f"峰值: {torch.cuda.max_memory_allocated(self.device)/1024**2:.1f}MB")


# 创建全局配置实例
config = Config()

