"""
评估指标模块
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_point_metrics(y_true, y_pred):
    """
    计算点预测指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        metrics: 指标字典
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (处理除零)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # R²
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def calculate_interval_metrics(y_true, lower, median, upper, target_coverage=0.8):
    """
    计算预测区间指标
    
    Args:
        y_true: 真实值
        lower: 下界
        median: 中位数
        upper: 上界
        target_coverage: 目标覆盖率
        
    Returns:
        metrics: 指标字典
    """
    n = len(y_true)
    
    # 1. PICP (Prediction Interval Coverage Probability)
    # 真实值落在区间内的比例
    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    
    # 2. MPIW (Mean Prediction Interval Width)
    # 预测区间的平均宽度
    interval_width = upper - lower
    mpiw = np.mean(interval_width)
    
    # 3. NMPIW (Normalized MPIW)
    # 归一化的区间宽度
    y_range = y_true.max() - y_true.min()
    nmpiw = mpiw / y_range if y_range > 0 else 0
    
    # 4. CWC (Coverage Width-based Criterion)
    # 综合考虑覆盖率和宽度的指标
    picp = coverage / 100
    eta = 50  # 惩罚系数
    gamma = 1 if picp < target_coverage else 0
    cwc = nmpiw * (1 + gamma * np.exp(-eta * (picp - target_coverage)))
    
    # 5. Winkler Score
    # 区间评分规则
    alpha = 1 - target_coverage
    winkler_scores = []
    for i in range(n):
        width = upper[i] - lower[i]
        penalty_lower = (2 / alpha) * (lower[i] - y_true[i]) if y_true[i] < lower[i] else 0
        penalty_upper = (2 / alpha) * (y_true[i] - upper[i]) if y_true[i] > upper[i] else 0
        score = width + penalty_lower + penalty_upper
        winkler_scores.append(score)
    winkler = np.mean(winkler_scores)
    
    # 6. Interval Score
    # 另一种区间评分规则
    interval_scores = []
    for i in range(n):
        width = upper[i] - lower[i]
        penalty_lower = (2 / alpha) * (lower[i] - y_true[i]) if y_true[i] < lower[i] else 0
        penalty_upper = (2 / alpha) * (y_true[i] - upper[i]) if y_true[i] > upper[i] else 0
        score = width + penalty_lower + penalty_upper
        interval_scores.append(score)
    interval_score = np.mean(interval_scores)
    
    # 7. ACE (Average Coverage Error)
    # 实际覆盖率与目标覆盖率的差异
    ace = abs(picp - target_coverage)
    
    # 8. PICP误差
    picp_error = abs(coverage - target_coverage * 100)
    
    # 点预测指标（使用中位数）
    point_metrics = calculate_point_metrics(y_true, median)
    
    return {
        'PICP': coverage,
        'PICP_Error': picp_error,
        'MPIW': mpiw,
        'NMPIW': nmpiw,
        'CWC': cwc,
        'Winkler': winkler,
        'IS': interval_score,
        'ACE': ace,
        **point_metrics
    }


def print_metrics(method_name, metrics):
    """
    打印指标
    
    Args:
        method_name: 方法名称
        metrics: 指标字典
    """
    print(f"\n{method_name} 评估结果:")
    print(f"  点预测指标:")
    print(f"    MAE: {metrics['MAE']:.4f}")
    print(f"    RMSE: {metrics['RMSE']:.4f}")
    print(f"    MAPE: {metrics['MAPE']:.2f}%")
    print(f"    R²: {metrics['R2']:.4f}")
    
    if 'PICP' in metrics:
        print(f"  区间预测指标:")
        print(f"    PICP: {metrics['PICP']:.2f}% (误差: {metrics['PICP_Error']:.2f}%)")
        print(f"    MPIW: {metrics['MPIW']:.4f}")
        print(f"    NMPIW: {metrics['NMPIW']:.4f}")
        print(f"    CWC: {metrics['CWC']:.4f} ↓")
        print(f"    Winkler Score: {metrics['Winkler']:.4f} ↓")
        print(f"    Interval Score: {metrics['IS']:.4f} ↓")
        print(f"    ACE: {metrics['ACE']:.4f} ↓")


def compare_methods(results_dict):
    """
    对比多个方法的结果
    
    Args:
        results_dict: {方法名: 指标字典}
        
    Returns:
        ranking: 排名结果
    """
    print("\n" + "="*80)
    print("方法对比排名")
    print("="*80)
    
    # 按CWC排名（越小越好）
    print("\n[综合性能排名 - 基于CWC]")
    methods_sorted = sorted(results_dict.items(), key=lambda x: x[1].get('CWC', float('inf')))
    for i, (method, metrics) in enumerate(methods_sorted, 1):
        symbol = "⭐" if method == "Adaptive_CAUN" else "  "
        if 'CWC' in metrics:
            print(f"  {symbol}{i}. {method:30s} CWC: {metrics['CWC']:.4f}")
    
    # 按PICP误差排名（越小越好）
    print("\n[覆盖率准确性排名 - 基于PICP误差]")
    methods_sorted = sorted(results_dict.items(), key=lambda x: x[1].get('PICP_Error', float('inf')))
    for i, (method, metrics) in enumerate(methods_sorted, 1):
        symbol = "⭐" if method == "Adaptive_CAUN" else "  "
        if 'PICP' in metrics:
            print(f"  {symbol}{i}. {method:30s} PICP: {metrics['PICP']:.2f}% (误差: {metrics['PICP_Error']:.2f}%)")
    
    # 按点预测MAE排名
    print("\n[点预测性能排名 - 基于MAE]")
    methods_sorted = sorted(results_dict.items(), key=lambda x: x[1].get('MAE', float('inf')))
    for i, (method, metrics) in enumerate(methods_sorted, 1):
        symbol = "⭐" if method == "Adaptive_CAUN" else "  "
        print(f"  {symbol}{i}. {method:30s} MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")

