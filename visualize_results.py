"""
可视化最终对比实验结果
生成论文级别的对比图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns

print("="*70)
print("生成最终对比实验可视化")
print("="*70)

# 读取数据
try:
    metrics_df = pd.read_csv('final_methods_comparison.csv')
    results_df = pd.read_csv('final_methods_predictions.csv')
    print(f"✓ 成功读取数据")
    print(f"  方法数: {len(metrics_df)}")
    print(f"  样本数: {len(results_df)}")
except FileNotFoundError:
    print("✗ 未找到结果文件，请先运行: python final_comparison.py")
    exit(1)

# ==================== 图1: 综合指标对比（柱状图）====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

methods = metrics_df['Method'].tolist()
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

metrics_to_plot = ['PICP', 'MPIW', 'MAE', 'RMSE', 'CWC', 'IS']
titles = [
    'PICP (Coverage %)\n目标: 80%', 
    'MPIW (Interval Width)\n越小越好', 
    'MAE (Point Accuracy)\n越小越好',
    'RMSE\n越小越好', 
    'CWC (综合指标)\n越小越好', 
    'IS (Interval Score)\n越小越好'
]

for idx, (ax, metric, title) in enumerate(zip(axes.flatten(), metrics_to_plot, titles)):
    values = metrics_df[metric].values
    bars = ax.bar(range(len(methods)), values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # 高亮Adaptive CAUN
    caun_idx = methods.index('Adaptive_CAUN') if 'Adaptive_CAUN' in methods else -1
    if caun_idx >= 0:
        bars[caun_idx].set_edgecolor('red')
        bars[caun_idx].set_linewidth(3)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 特殊标记
    if metric == 'PICP':
        ax.axhline(y=80, color='green', linestyle='--', linewidth=2, label='Target: 80%', alpha=0.7)
        ax.legend(fontsize=9)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=0, fontsize=9)
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('final_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ 综合指标对比图: final_metrics_comparison.png")
plt.close()

# ==================== 图2: 预测区间可视化（前150个点）====================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

methods_with_intervals = [m for m in methods if f'{m}_Median' in results_df.columns]
time_steps = np.arange(min(150, len(results_df)))
true_values = results_df['True_Value'].values[:150]

for idx, (ax, method) in enumerate(zip(axes, methods_with_intervals)):
    if idx >= len(axes):
        break
    
    median = results_df[f'{method}_Median'].values[:150]
    lower = results_df[f'{method}_Lower'].values[:150]
    upper = results_df[f'{method}_Upper'].values[:150]
    
    # 计算覆盖率
    covered = (true_values >= lower) & (true_values <= upper)
    coverage = np.mean(covered) * 100
    
    # 绘图
    ax.plot(time_steps, true_values, 'k-', linewidth=2.5, label='True Value', alpha=0.8, zorder=4)
    ax.plot(time_steps, median, color=colors[idx], linewidth=2, 
           label='Prediction', alpha=0.7, zorder=3)
    ax.fill_between(time_steps, lower, upper, alpha=0.25, color=colors[idx],
                   label='80% PI', zorder=2)
    
    # 标记未覆盖点
    uncovered_idx = time_steps[~covered]
    if len(uncovered_idx) > 0:
        ax.scatter(uncovered_idx, true_values[~covered], color='red', s=30, 
                  marker='x', linewidth=2, label='Uncovered', zorder=5)
    
    ax.set_title(f'{method.replace("_", " ")}\nPICP: {coverage:.1f}%', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=10, fontweight='bold')
    ax.set_ylabel('Power (MW)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_intervals_visualization.png', dpi=300, bbox_inches='tight')
print("✓ 预测区间可视化: final_intervals_visualization.png")
plt.close()

# ==================== 图3: 综合排名（水平柱状图）====================
fig, ax = plt.subplots(figsize=(14, 8))

# 按CWC排序
metrics_df_sorted = metrics_df.sort_values('CWC')

y_pos = np.arange(len(metrics_df_sorted))
cwc_values = metrics_df_sorted['CWC'].values

bars = ax.barh(y_pos, cwc_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# 高亮Adaptive CAUN
for i, method in enumerate(metrics_df_sorted['Method']):
    if method == 'Adaptive_CAUN':
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(3)
        bars[i].set_alpha(0.9)

ax.set_yticks(y_pos)
ax.set_yticklabels(metrics_df_sorted['Method'].str.replace('_', ' '), fontsize=12, fontweight='bold')
ax.set_xlabel('CWC (Coverage Width Criterion) - Lower is Better', fontsize=13, fontweight='bold')
ax.set_title('Overall Ranking of Uncertainty Quantification Methods\n(Based on CWC)', 
            fontsize=15, fontweight='bold', pad=20)

# 添加数值和排名标签
for idx, (bar, cwc, method) in enumerate(zip(bars, cwc_values, metrics_df_sorted['Method'])):
    width = bar.get_width()
    symbol = '⭐' if method == 'Adaptive_CAUN' else ''
    ax.text(width, bar.get_y() + bar.get_height()/2,
           f' {symbol}#{idx+1}: {cwc:.2f}', ha='left', va='center', 
           fontsize=11, fontweight='bold')

ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('final_ranking.png', dpi=300, bbox_inches='tight')
print("✓ 综合排名图: final_ranking.png")
plt.close()

# ==================== 图4: 详细指标表格（论文用）====================
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')

# 准备表格数据
table_data = []
columns = ['Rank', 'Method', 'PICP (%)', 'MPIW', 'MAE', 'RMSE', 'CWC', 'IS']

metrics_df_sorted = metrics_df.sort_values('CWC')
for idx, row in enumerate(metrics_df_sorted.itertuples(), 1):
    symbol = '⭐' if row.Method == 'Adaptive_CAUN' else ''
    table_row = [
        f'{symbol}#{idx}',
        row.Method.replace('_', ' '),
        f"{row.PICP:.2f}",
        f"{row.MPIW:.3f}",
        f"{row.MAE:.3f}",
        f"{row.RMSE:.3f}",
        f"{row.CWC:.3f}",
        f"{row.IS:.3f}"
    ]
    table_data.append(table_row)

table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center',
                loc='center', bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# 设置表头样式
for i in range(len(columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(weight='bold', color='white')

# 设置行颜色
for i in range(len(table_data)):
    for j in range(len(columns)):
        cell = table[(i+1, j)]
        if j == 0:  # Rank列
            cell.set_facecolor('#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32' if i == 2 else '#F0F0F0')
            cell.set_text_props(weight='bold')
        elif '⭐' in str(table_data[i][0]):  # Adaptive CAUN行
            cell.set_facecolor('#FFE5E5')
            if j == 1:
                cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F8F8F8')

plt.title('Comprehensive Comparison of Uncertainty Quantification Methods',
         fontsize=14, fontweight='bold', pad=20)
plt.savefig('final_comparison_table.png', dpi=300, bbox_inches='tight')
print("✓ 详细对比表格: final_comparison_table.png")
plt.close()

# ==================== 打印总结 ====================
print("\n" + "="*70)
print("结果总结")
print("="*70)

print("\n🏆 按CWC排名 (越小越好):")
for idx, row in enumerate(metrics_df_sorted.itertuples(), 1):
    symbol = "⭐" if row.Method == 'Adaptive_CAUN' else "  "
    print(f"  {symbol}{idx}. {row.Method:25s} - CWC: {row.CWC:.4f}, PICP: {row.PICP:.2f}%")

print("\n📊 按PICP最接近80%排名:")
metrics_df['PICP_Error'] = np.abs(metrics_df['PICP'] - 80)
metrics_df_picp = metrics_df.sort_values('PICP_Error')
for idx, row in enumerate(metrics_df_picp.itertuples(), 1):
    symbol = "⭐" if row.Method == 'Adaptive_CAUN' else "  "
    print(f"  {symbol}{idx}. {row.Method:25s} - PICP: {row.PICP:.2f}% (误差: {row.PICP_Error:.2f}%)")

print("\n🎯 按MAE排名 (越小越好):")
metrics_df_mae = metrics_df.sort_values('MAE')
for idx, row in enumerate(metrics_df_mae.itertuples(), 1):
    symbol = "⭐" if row.Method == 'Adaptive_CAUN' else "  "
    print(f"  {symbol}{idx}. {row.Method:25s} - MAE: {row.MAE:.4f}")

print("\n" + "="*70)
print("所有可视化已生成！")
print("="*70)
print("\n生成的文件:")
print("  1. final_metrics_comparison.png    - 综合指标对比")
print("  2. final_intervals_visualization.png - 预测区间可视化")
print("  3. final_ranking.png                - 综合排名")
print("  4. final_comparison_table.png       - 详细对比表格（论文用）")
print("="*70)

