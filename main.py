"""
主运行脚本 - 统一管理所有模型的训练和评估
"""
import os
import sys

# OpenMP设置（避免多线程冲突）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# PyTorch线程设置
torch.set_num_threads(1)

# 导入配置和工具
from config import config
from data_utils import (
    load_and_preprocess_data,
    create_dataloader,
    inverse_transform_predictions
)
from metrics import calculate_interval_metrics, print_metrics, compare_methods

# 导入模型
from models import (
    BaselinePointModel,
    QuantileRegressionModel,
    GPApproximationModel,
    NPKDEUncertainty,
    AdaptiveCAUN,
    UncertaintyDetector
)
from models.quantile_regression import quantile_loss
from models.gp_approximation import gaussian_nll_loss
from models.adaptive_caun import adaptive_loss

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# OpenMP设置（避免多线程冲突）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_baseline_model(model, dataloader, config):
    """训练基准模型"""
    print("\n" + "="*80)
    print("训练基准点预测模型")
    print("="*80)
    
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
            config.print_gpu_stats()
    
    # 保存模型
    torch.save(model.state_dict(), f'{config.checkpoint_dir}/baseline_model.pt')
    print("  ✓ 模型已保存")
    
    return model


def train_quantile_model(model, dataloader, config):
    """训练分位数回归模型"""
    print("\n" + "="*80)
    print("训练分位数回归模型")
    print("="*80)
    
    config.clear_gpu_cache()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    model.train()
    for epoch in range(config.epochs_quantile):
        epoch_loss = 0
        for data, target in dataloader:
            data = data.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True).squeeze()
            
            optimizer.zero_grad()
            pred = model(data)
            loss = quantile_loss(pred, target, config.quantiles)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{config.epochs_quantile}, Loss: {avg_loss:.6f}")
            config.print_gpu_stats()
    
    torch.save(model.state_dict(), f'{config.checkpoint_dir}/quantile_model.pt')
    print("  ✓ 模型已保存")
    
    return model


def train_gp_model(model, dataloader, config):
    """训练高斯过程近似模型"""
    print("\n" + "="*80)
    print("训练高斯过程近似模型")
    print("="*80)
    
    config.clear_gpu_cache()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    model.train()
    for epoch in range(config.epochs_gp):
        epoch_loss = 0
        for data, target in dataloader:
            data = data.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True).squeeze()
            
            optimizer.zero_grad()
            mean, var = model(data)
            loss = gaussian_nll_loss(mean, var, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{config.epochs_gp}, Loss: {avg_loss:.6f}")
            config.print_gpu_stats()
    
    torch.save(model.state_dict(), f'{config.checkpoint_dir}/gp_model.pt')
    print("  ✓ 模型已保存")
    
    return model


def train_adaptive_caun(model, dataloader, uncertainty_scores, config):
    """训练自适应CAUN模型"""
    print("\n" + "="*80)
    print("训练Adaptive CAUN模型 (我们的创新方法) ⭐")
    print("="*80)
    
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
    
    torch.save(model.state_dict(), f'{config.checkpoint_dir}/adaptive_caun_model.pt')
    print("  ✓ 模型已保存")
    
    return model


def main():
    """主函数"""
    print("="*80)
    print("多特征光伏发电不确定性量化实验 (GPU优化版)")
    print("="*80)
    print("\n对比方法:")
    print("  1. Baseline Point Prediction (无不确定性)")
    print("  2. Quantile Regression (直接分位数预测)")
    print("  3. GP Approximation (高斯过程近似)")
    print("  4. NPKDE (传统核密度估计)")
    print("  5. Adaptive CAUN (我们的创新方法) ⭐")
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
    print(f"  训练集: {len(X_train)} 样本, {len(dataloader_train)} batches")
    print(f"  测试集: {len(X_test)} 样本, {len(dataloader_test)} batches")
    
    # 反归一化测试集真实值
    y_test_original = inverse_transform_predictions(
        y_test.numpy().flatten(), scaler, data_dim
    )
    
    # 存储所有结果
    all_results = {}
    
    # ==================== 方法1: Baseline ====================
    print("\n\n" + "="*80)
    print("方法1: Baseline Point Prediction")
    print("="*80)
    
    baseline_model = BaselinePointModel(
        data_dim, config.n_filters, config.n_filters1, 
        config.filter_size, config.dropout_rate
    ).to(config.device)
    
    baseline_model = train_baseline_model(baseline_model, dataloader_train, config)
    
    # 测试
    baseline_model.eval()
    with torch.no_grad():
        baseline_pred = baseline_model(X_test.to(config.device)).cpu().numpy()
    
    baseline_pred_original = inverse_transform_predictions(
        baseline_pred, scaler, data_dim
    )
    
    # 计算点预测指标
    from metrics import calculate_point_metrics
    baseline_metrics = calculate_point_metrics(y_test_original, baseline_pred_original)
    print_metrics("Baseline Point Prediction", baseline_metrics)
    
    all_results['Baseline'] = {
        'predictions': baseline_pred_original,
        'metrics': baseline_metrics
    }
    
    # ==================== 方法2: Quantile Regression ====================
    print("\n\n" + "="*80)
    print("方法2: Quantile Regression")
    print("="*80)
    
    qr_model = QuantileRegressionModel(
        data_dim, config.n_filters, config.n_filters1,
        config.filter_size, config.quantiles, config.dropout_rate
    ).to(config.device)
    
    qr_model = train_quantile_model(qr_model, dataloader_train, config)
    
    # 测试
    qr_model.eval()
    with torch.no_grad():
        qr_lower, qr_median, qr_upper = qr_model.predict_with_uncertainty(
            X_test.to(config.device)
        )
    
    qr_lower_original = inverse_transform_predictions(qr_lower.cpu().numpy(), scaler, data_dim)
    qr_median_original = inverse_transform_predictions(qr_median.cpu().numpy(), scaler, data_dim)
    qr_upper_original = inverse_transform_predictions(qr_upper.cpu().numpy(), scaler, data_dim)
    
    qr_metrics = calculate_interval_metrics(
        y_test_original, qr_lower_original, qr_median_original, 
        qr_upper_original, config.target_coverage
    )
    print_metrics("Quantile Regression", qr_metrics)
    
    all_results['Quantile_Regression'] = {
        'lower': qr_lower_original,
        'median': qr_median_original,
        'upper': qr_upper_original,
        'metrics': qr_metrics
    }
    
    # ==================== 方法3: GP Approximation ====================
    print("\n\n" + "="*80)
    print("方法3: GP Approximation")
    print("="*80)
    
    gp_model = GPApproximationModel(
        data_dim, config.n_filters, config.n_filters1,
        config.filter_size, config.dropout_rate
    ).to(config.device)
    
    gp_model = train_gp_model(gp_model, dataloader_train, config)
    
    # 测试
    gp_model.eval()
    with torch.no_grad():
        gp_lower, gp_median, gp_upper = gp_model.predict_with_uncertainty(
            X_test.to(config.device), confidence=config.target_coverage
        )
    
    gp_lower_original = inverse_transform_predictions(gp_lower.cpu().numpy(), scaler, data_dim)
    gp_median_original = inverse_transform_predictions(gp_median.cpu().numpy(), scaler, data_dim)
    gp_upper_original = inverse_transform_predictions(gp_upper.cpu().numpy(), scaler, data_dim)
    
    gp_metrics = calculate_interval_metrics(
        y_test_original, gp_lower_original, gp_median_original,
        gp_upper_original, config.target_coverage
    )
    print_metrics("GP Approximation", gp_metrics)
    
    all_results['GP_Approximation'] = {
        'lower': gp_lower_original,
        'median': gp_median_original,
        'upper': gp_upper_original,
        'metrics': gp_metrics
    }
    
    # ==================== 方法4: NPKDE ====================
    print("\n\n" + "="*80)
    print("方法4: NPKDE (传统方法)")
    print("="*80)
    
    # 获取训练集预测用于拟合NPKDE
    baseline_model.eval()
    with torch.no_grad():
        train_pred = baseline_model(X_train.to(config.device)).cpu().numpy()
    train_pred_original = inverse_transform_predictions(train_pred, scaler, data_dim)
    
    # 训练集真实值
    y_train_original = inverse_transform_predictions(
        y_train.numpy().flatten(), scaler, data_dim
    )
    
    # 拟合NPKDE
    npkde = NPKDEUncertainty(train_pred_original, y_train_original, bandwidth='scott')
    
    # 测试
    npkde_lower, npkde_median, npkde_upper = npkde.predict_intervals(
        baseline_pred_original, n_samples=1000, confidence=config.target_coverage
    )
    
    npkde_metrics = calculate_interval_metrics(
        y_test_original, npkde_lower, npkde_median,
        npkde_upper, config.target_coverage
    )
    print_metrics("NPKDE", npkde_metrics)
    
    all_results['NPKDE'] = {
        'lower': npkde_lower,
        'median': npkde_median,
        'upper': npkde_upper,
        'metrics': npkde_metrics
    }
    
    # ==================== 方法5: Adaptive CAUN ⭐ ====================
    print("\n\n" + "="*80)
    print("方法5: Adaptive CAUN (我们的创新方法) ⭐")
    print("="*80)
    
    # 阶段1: 识别高不确定性区域
    print("\n[阶段1: 识别高不确定性区域]")
    detector = UncertaintyDetector(threshold_percentile=config.uncertainty_threshold_percentile)
    high_unc_mask, unc_scores = detector.detect(y_test_original, baseline_pred_original)
    
    # 阶段2: 训练Adaptive CAUN
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
    
    # 测试
    adaptive_caun.eval()
    with torch.no_grad():
        caun_lower, caun_median, caun_upper = adaptive_caun.predict_with_uncertainty(
            X_test.to(config.device), unc_scores
        )
    
    caun_lower_original = inverse_transform_predictions(caun_lower.cpu().numpy(), scaler, data_dim)
    caun_median_original = inverse_transform_predictions(caun_median.cpu().numpy(), scaler, data_dim)
    caun_upper_original = inverse_transform_predictions(caun_upper.cpu().numpy(), scaler, data_dim)
    
    caun_metrics = calculate_interval_metrics(
        y_test_original, caun_lower_original, caun_median_original,
        caun_upper_original, config.target_coverage
    )
    print_metrics("Adaptive CAUN", caun_metrics)
    
    all_results['Adaptive_CAUN'] = {
        'lower': caun_lower_original,
        'median': caun_median_original,
        'upper': caun_upper_original,
        'metrics': caun_metrics,
        'uncertainty_scores': unc_scores
    }
    
    # ==================== 结果汇总 ====================
    print("\n\n" + "="*80)
    print("实验完成！")
    print("="*80)
    
    # 对比所有方法
    metrics_dict = {k: v['metrics'] for k, v in all_results.items() if 'metrics' in v}
    compare_methods(metrics_dict)
    
    # 保存结果
    print("\n[保存结果]")
    
    # 保存指标
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.to_csv(f'{config.results_dir}/all_methods_metrics.csv')
    print(f"  ✓ 指标已保存: {config.results_dir}/all_methods_metrics.csv")
    
    # 保存预测结果
    predictions_data = {
        'y_true': y_test_original
    }
    for method, results in all_results.items():
        if 'median' in results:
            predictions_data[f'{method}_median'] = results['median']
            predictions_data[f'{method}_lower'] = results['lower']
            predictions_data[f'{method}_upper'] = results['upper']
        elif 'predictions' in results:
            predictions_data[f'{method}_pred'] = results['predictions']
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv(f'{config.results_dir}/all_methods_predictions.csv', index=False)
    print(f"  ✓ 预测结果已保存: {config.results_dir}/all_methods_predictions.csv")
    
    # GPU使用总结
    if torch.cuda.is_available():
        print("\n[GPU使用总结]")
        print(f"  设备: {torch.cuda.get_device_name(0)}")
        print(f"  峰值内存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        print(f"  当前内存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    config.clear_gpu_cache()
    
    print("\n" + "="*80)
    print("所有结果已保存在 results/ 目录下")
    print("="*80)
    
    return all_results, metrics_df


if __name__ == "__main__":
    results, metrics = main()

