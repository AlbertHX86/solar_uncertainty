"""
仅测试Adaptive CAUN方法
快速验证我们的创新方法
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# OpenMP设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# 导入模块
from config import config
from data_utils import (
    load_and_preprocess_data,
    create_dataloader,
    inverse_transform_predictions
)
from metrics import calculate_interval_metrics, print_metrics

from models import BaselinePointModel, AdaptiveCAUN, UncertaintyDetector
from models.adaptive_caun import adaptive_loss

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


def train_baseline_for_caun(model, dataloader, config):
    """训练基准模型（为CAUN提供初始预测）"""
    print("\n[步骤1/3] 训练基准点预测模型...")
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(config.epochs_baseline):
        epoch_loss = 0
        for data, target in dataloader:
            data = data.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True).squeeze()
            
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{config.epochs_baseline}, Loss: {avg_loss:.6f}")
    
    print("  ✓ 基准模型训练完成")
    return model


def train_adaptive_caun(model, dataloader, uncertainty_scores, config):
    """训练Adaptive CAUN模型"""
    print("\n[步骤3/3] 训练Adaptive CAUN...")
    
    config.clear_gpu_cache()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 归一化不确定性分数
    unc_scores_norm = (uncertainty_scores - uncertainty_scores.min()) / \
                      (uncertainty_scores.max() - uncertainty_scores.min() + 1e-8)
    
    model.train()
    for epoch in range(config.epochs_caun_stage2):
        epoch_loss = 0
        
        for data, target in dataloader:
            data = data.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True).squeeze()
            
            # 为当前batch采样不确定性分数
            batch_size = len(target)
            batch_unc_scores = np.random.choice(unc_scores_norm, size=batch_size)
            
            optimizer.zero_grad()
            predictions = model(data, batch_unc_scores)
            loss = adaptive_loss(predictions, target, batch_unc_scores)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{config.epochs_caun_stage2}, Loss: {avg_loss:.6f}")
            config.print_gpu_stats()
    
    print("  ✓ Adaptive CAUN训练完成")
    return model


def visualize_results(y_true, lower, median, upper, save_path='results/caun_results.png'):
    """可视化预测结果"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 第1天（前45个点）
    ax1 = axes[0]
    x1 = np.arange(45)
    ax1.plot(x1, y_true[:45], 'k-', label='True Values', linewidth=2, alpha=0.8)
    ax1.plot(x1, median[:45], 'b-', label='CAUN Prediction', linewidth=2)
    ax1.fill_between(x1, lower[:45], upper[:45], alpha=0.3, color='blue', label='80% Interval')
    ax1.set_xlabel('Time Step (15-min intervals)', fontsize=12)
    ax1.set_ylabel('Power (MW)', fontsize=12)
    ax1.set_title('Day 1: Adaptive CAUN Prediction', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 第2天（后45个点）
    ax2 = axes[1]
    x2 = np.arange(45)
    ax2.plot(x2, y_true[45:], 'k-', label='True Values', linewidth=2, alpha=0.8)
    ax2.plot(x2, median[45:], 'b-', label='CAUN Prediction', linewidth=2)
    ax2.fill_between(x2, lower[45:], upper[45:], alpha=0.3, color='blue', label='80% Interval')
    ax2.set_xlabel('Time Step (15-min intervals)', fontsize=12)
    ax2.set_ylabel('Power (MW)', fontsize=12)
    ax2.set_title('Day 2: Adaptive CAUN Prediction', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ 可视化结果已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("="*80)
    print("Adaptive CAUN 单独测试")
    print("="*80)
    print("\n我们的创新方法：")
    print("  - Cross-Attention机制（Transformer Decoder）")
    print("  - 两阶段训练策略")
    print("  - 自适应区间调整")
    print("="*80)
    
    # 创建必要的目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    # ==================== 数据准备 ====================
    data_train, data_test, scaler, feature_names, data_dim = load_and_preprocess_data(config)
    
    # 创建DataLoader
    dataloader_train, X_train, y_train = create_dataloader(
        data_train, config.window, config.predict_length, 
        config.batch_size, data_dim, shuffle=True
    )
    dataloader_test, X_test, y_test = create_dataloader(
        data_test, config.window, config.predict_length,
        config.batch_size, data_dim, shuffle=False
    )
    
    print(f"\n[DataLoader准备完成]")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 反归一化测试集真实值
    y_test_original = inverse_transform_predictions(
        y_test.numpy().flatten(), scaler, data_dim
    )
    
    # ==================== 阶段1: 训练基准模型 ====================
    baseline_model = BaselinePointModel(
        data_dim, config.n_filters, config.n_filters1, 
        config.filter_size, config.dropout_rate
    ).to(config.device)
    
    baseline_model = train_baseline_for_caun(baseline_model, dataloader_train, config)
    
    # 获取基准预测
    baseline_model.eval()
    with torch.no_grad():
        baseline_pred = baseline_model(X_test.to(config.device)).cpu().numpy()
    
    baseline_pred_original = inverse_transform_predictions(
        baseline_pred, scaler, data_dim
    )
    
    # ==================== 阶段2: 识别高不确定性区域 ====================
    print("\n[步骤2/3] 识别高不确定性区域...")
    detector = UncertaintyDetector(threshold_percentile=config.uncertainty_threshold_percentile)
    high_unc_mask, unc_scores = detector.detect(y_test_original, baseline_pred_original)
    
    # ==================== 阶段3: 训练Adaptive CAUN ====================
    adaptive_caun = AdaptiveCAUN(
        base_encoder=baseline_model,
        decoder_dim=config.decoder_dim,
        num_heads=config.num_heads,
        num_decoder_layers=config.num_decoder_layers,
        dropout_rate=config.dropout_rate
    ).to(config.device)
    
    adaptive_caun = train_adaptive_caun(
        adaptive_caun, dataloader_train, unc_scores, config
    )
    
    # 保存模型
    torch.save(adaptive_caun.state_dict(), f'{config.checkpoint_dir}/adaptive_caun_only.pt')
    print(f"  ✓ 模型已保存: {config.checkpoint_dir}/adaptive_caun_only.pt")
    
    # ==================== 测试和评估 ====================
    print("\n" + "="*80)
    print("测试和评估")
    print("="*80)
    
    adaptive_caun.eval()
    with torch.no_grad():
        caun_lower, caun_median, caun_upper = adaptive_caun.predict_with_uncertainty(
            X_test.to(config.device), unc_scores
        )
    
    caun_lower_original = inverse_transform_predictions(caun_lower.cpu().numpy(), scaler, data_dim)
    caun_median_original = inverse_transform_predictions(caun_median.cpu().numpy(), scaler, data_dim)
    caun_upper_original = inverse_transform_predictions(caun_upper.cpu().numpy(), scaler, data_dim)
    
    # 计算指标
    caun_metrics = calculate_interval_metrics(
        y_test_original, caun_lower_original, caun_median_original,
        caun_upper_original, config.target_coverage
    )
    print_metrics("Adaptive CAUN", caun_metrics)
    
    # ==================== 保存结果 ====================
    print("\n" + "="*80)
    print("保存结果")
    print("="*80)
    
    # 保存指标
    metrics_df = pd.DataFrame([caun_metrics], index=['Adaptive_CAUN'])
    metrics_df.to_csv(f'{config.results_dir}/caun_only_metrics.csv')
    print(f"  ✓ 指标已保存: {config.results_dir}/caun_only_metrics.csv")
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        'y_true': y_test_original,
        'CAUN_lower': caun_lower_original,
        'CAUN_median': caun_median_original,
        'CAUN_upper': caun_upper_original,
        'uncertainty_score': unc_scores,
        'high_uncertainty': high_unc_mask
    })
    predictions_df.to_csv(f'{config.results_dir}/caun_only_predictions.csv', index=False)
    print(f"  ✓ 预测结果已保存: {config.results_dir}/caun_only_predictions.csv")
    
    # 可视化
    visualize_results(
        y_test_original, 
        caun_lower_original, 
        caun_median_original, 
        caun_upper_original,
        save_path=f'{config.results_dir}/caun_only_visualization.png'
    )
    
    # GPU使用总结
    if torch.cuda.is_available():
        print("\n[GPU使用总结]")
        print(f"  设备: {torch.cuda.get_device_name(0)}")
        print(f"  峰值内存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    config.clear_gpu_cache()
    
    print("\n" + "="*80)
    print("✓ 测试完成！")
    print("="*80)
    print(f"\n关键指标：")
    print(f"  PICP: {caun_metrics['PICP']:.2f}% (目标: 80%)")
    print(f"  MAE: {caun_metrics['MAE']:.4f}")
    print(f"  CWC: {caun_metrics['CWC']:.4f} (越小越好)")
    print(f"\n结果文件：")
    print(f"  - {config.results_dir}/caun_only_metrics.csv")
    print(f"  - {config.results_dir}/caun_only_predictions.csv")
    print(f"  - {config.results_dir}/caun_only_visualization.png")
    
    return caun_metrics, predictions_df


if __name__ == "__main__":
    metrics, predictions = main()

