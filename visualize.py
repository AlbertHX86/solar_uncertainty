"""
可视化工具 - 生成实验结果图表
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style('whitegrid')


def plot_metrics_comparison(metrics_csv='results/all_methods_metrics.csv', 
                            save_path='results/metrics_comparison.png'):
    """对比不同方法的指标"""
    df = pd.read_csv(metrics_csv, index_col=0)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Uncertainty Quantification Methods Comparison', fontsize=16, fontweight='bold')
    
    # 定义指标和对应的排序方式
    metrics_to_plot = [
        ('MAE', 'minimize', 'Mean Absolute Error'),
        ('RMSE', 'minimize', 'Root Mean Squared Error'),
        ('PICP', 'target_80', 'Prediction Interval Coverage (%)'),
        ('MPIW', 'minimize', 'Mean Prediction Interval Width'),
        ('CWC', 'minimize', 'Coverage Width Criterion'),
        ('Winkler', 'minimize', 'Winkler Score')
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, (metric, order, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        if metric in df.columns:
            data = df[metric].sort_values()
            
            # 高亮Adaptive CAUN
            bar_colors = ['#9b59b6' if x == 'Adaptive_CAUN' else '#3498db' for x in data.index]
            
            bars = ax.barh(range(len(data)), data.values, color=bar_colors, alpha=0.8)
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data.index, fontsize=10)
            ax.set_xlabel(title, fontsize=11)
            ax.set_title(f'{title}', fontsize=12, fontweight='bold')
            
            # 添加数值标签
            for i, (val, name) in enumerate(zip(data.values, data.index)):
                symbol = ' ⭐' if name == 'Adaptive_CAUN' else ''
                ax.text(val, i, f' {val:.3f}{symbol}', va='center', fontsize=9)
            
            # 如果是PICP，添加目标线
            if metric == 'PICP':
                ax.axvline(x=80, color='red', linestyle='--', linewidth=2, label='Target (80%)')
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 指标对比图已保存: {save_path}")
    plt.close()


def plot_interval_visualization(predictions_csv='results/all_methods_predictions.csv',
                                save_path='results/intervals_visualization.png',
                                n_samples=200):
    """可视化预测区间"""
    df = pd.read_csv(predictions_csv)
    
    # 只显示前n_samples个样本
    df = df.head(n_samples)
    x = np.arange(len(df))
    
    methods = ['Quantile_Regression', 'GP_Approximation', 'NPKDE', 'Adaptive_CAUN']
    titles = ['Quantile Regression', 'GP Approximation', 'NPKDE', 'Adaptive CAUN ⭐']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, (method, title) in enumerate(zip(methods, titles)):
        ax = axes[idx]
        
        # 绘制真实值
        ax.plot(x, df['y_true'], 'k-', label='True Values', linewidth=1.5, alpha=0.7)
        
        # 绘制预测值
        ax.plot(x, df[f'{method}_median'], 'b-', label='Prediction', linewidth=1.5)
        
        # 绘制预测区间
        ax.fill_between(x, df[f'{method}_lower'], df[f'{method}_upper'],
                        alpha=0.3, color='blue', label='80% Prediction Interval')
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Power (MW)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 区间可视化已保存: {save_path}")
    plt.close()


def plot_detailed_comparison(predictions_csv='results/all_methods_predictions.csv',
                             save_path='results/detailed_comparison.png',
                             start_idx=0, end_idx=100):
    """详细对比图（单个方法放大）"""
    df = pd.read_csv(predictions_csv)
    df = df.iloc[start_idx:end_idx]
    x = np.arange(len(df))
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    
    methods = [
        ('Baseline', 'Baseline Point Prediction', 'pred'),
        ('Quantile_Regression', 'Quantile Regression', 'median'),
        ('GP_Approximation', 'GP Approximation', 'median'),
        ('NPKDE', 'NPKDE', 'median'),
        ('Adaptive_CAUN', 'Adaptive CAUN ⭐', 'median')
    ]
    
    for idx, (method, title, pred_type) in enumerate(methods):
        ax = axes[idx]
        
        # 真实值
        ax.plot(x, df['y_true'], 'k-', label='True Values', linewidth=2, alpha=0.8)
        
        # 预测值
        if pred_type == 'pred':
            ax.plot(x, df[f'{method}_pred'], 'r--', label='Point Prediction', linewidth=1.5)
        else:
            ax.plot(x, df[f'{method}_median'], 'b-', label='Median Prediction', linewidth=1.5)
            
            # 预测区间
            ax.fill_between(x, df[f'{method}_lower'], df[f'{method}_upper'],
                           alpha=0.3, color='blue', label='80% Interval')
        
        ax.set_ylabel('Power (MW)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Step', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 详细对比图已保存: {save_path}")
    plt.close()


def plot_ranking_table(metrics_csv='results/all_methods_metrics.csv',
                       save_path='results/ranking_table.png'):
    """生成排名表格"""
    df = pd.read_csv(metrics_csv, index_col=0)
    
    # 选择关键指标
    key_metrics = ['MAE', 'RMSE', 'PICP', 'PICP_Error', 'CWC', 'Winkler']
    df_display = df[key_metrics].copy()
    
    # 计算排名
    rankings = {}
    for col in key_metrics:
        if col == 'PICP':
            # PICP越接近80越好
            rankings[col] = df_display[col].apply(lambda x: abs(x - 80)).rank().astype(int)
        else:
            # 其他指标越小越好
            rankings[col] = df_display[col].rank().astype(int)
    
    rankings_df = pd.DataFrame(rankings)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # 子图1: 指标值
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=df_display.round(3).values,
                      rowLabels=df_display.index,
                      colLabels=df_display.columns,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.15] * len(df_display.columns))
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2)
    ax1.set_title('Metrics Values', fontsize=14, fontweight='bold', pad=20)
    
    # 子图2: 排名
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=rankings_df.values,
                      rowLabels=rankings_df.index,
                      colLabels=rankings_df.columns,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.15] * len(rankings_df.columns))
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)
    ax2.set_title('Rankings (1=Best)', fontsize=14, fontweight='bold', pad=20)
    
    # 高亮Adaptive_CAUN行
    for key, cell in table1.get_celld().items():
        if key[0] > 0:  # 跳过表头
            if df_display.index[key[0]-1] == 'Adaptive_CAUN':
                cell.set_facecolor('#E6E6FA')
                cell.set_text_props(weight='bold')
    
    for key, cell in table2.get_celld().items():
        if key[0] > 0:
            if rankings_df.index[key[0]-1] == 'Adaptive_CAUN':
                cell.set_facecolor('#E6E6FA')
                cell.set_text_props(weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 排名表格已保存: {save_path}")
    plt.close()


def main():
    """生成所有可视化"""
    print("\n" + "="*80)
    print("生成可视化结果")
    print("="*80)
    
    try:
        plot_metrics_comparison()
        plot_interval_visualization()
        plot_detailed_comparison()
        plot_ranking_table()
        
        print("\n" + "="*80)
        print("所有可视化已完成！")
        print("="*80)
    except Exception as e:
        print(f"错误: {e}")
        print("请确保已运行 main.py 并生成了结果文件")


if __name__ == "__main__":
    main()

