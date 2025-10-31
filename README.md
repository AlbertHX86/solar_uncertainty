# 多特征光伏发电不确定性量化

基于TCN-BiLSTM的光伏发电不确定性量化研究，对比5种方法。

## 方法说明

### 1. Baseline Point Prediction
基准点预测模型，仅提供点预测，不估计不确定性。

### 2. Quantile Regression
分位数回归模型，直接预测10%、50%、90%分位数。

### 3. GP Approximation  
高斯过程近似模型，预测均值和方差，基于高斯分布假设生成预测区间。

### 4. NPKDE
非参数核密度估计（传统方法），基于残差分布建模不确定性。

### 5. Adaptive CAUN ⭐（我们的方法）
**自适应交叉注意力不确定性网络**

**创新点**：
- 使用Transformer Decoder的Cross-Attention机制建模不确定性
- 两阶段训练：先识别高不确定性区域，再针对性建模
- 自适应调整预测区间宽度（高不确定性→宽区间，低不确定性→窄区间）
- 自适应损失函数，结合分位数损失、区间合理性约束和宽度惩罚

**架构**：
```
输入特征 → TCN-BiLSTM编码器 → Transformer Decoder (Cross-Attention) 
        → 分位数预测头 → 不确定性权重网络 → 自适应区间调整
```

## 数据说明

- **输入特征**（7维）：
  - Total solar irradiance (W/m²)
  - Direct normal irradiance (W/m²)  
  - Global horizontal irradiance (W/m²)
  - Air temperature (°C)
  - Atmosphere (hpa)
  - Relative humidity (%)
  - Power (MW) - 目标变量

- **数据规模**：32,895样本，15分钟间隔
- **训练/测试划分**：70% / 30%

## 训练流程

### 安装依赖
```bash
pip install -r requirements.txt
```

### 快速测试（可选）
```bash
python quick_test.py  # 2 epochs，约3-5分钟
```

### 完整训练
```bash
python main.py
```

**训练过程**：
1. 加载和预处理数据（power.xlsx）
2. 训练Baseline模型（30 epochs）
3. 训练Quantile Regression模型（30 epochs）
4. 训练GP Approximation模型（30 epochs）
5. 训练NPKDE（基于Baseline的残差）
6. 两阶段训练Adaptive CAUN：
   - 阶段1：使用Baseline结果识别高不确定性区域
   - 阶段2：训练CAUN模型（30 epochs）
7. 评估所有模型，计算12个指标
8. 保存结果到`results/`目录

**训练时间**：约20-30分钟（GPU）

### 生成可视化
```bash
python visualize.py
```

生成4种图表：
- `metrics_comparison.png` - 指标对比
- `intervals_visualization.png` - 预测区间可视化
- `detailed_comparison.png` - 详细对比
- `ranking_table.png` - 排名表格

## 评估指标

**点预测指标**：MAE, RMSE, MAPE, R²

**区间预测指标**：
- PICP (覆盖率) - 目标80%
- MPIW (平均区间宽度) - 越小越好
- CWC (综合指标) - 越小越好
- Winkler Score - 越小越好
- Interval Score - 越小越好
- ACE (覆盖误差) - 越小越好

## 调整参数

编辑 `config.py` 修改超参数：
```python
# 训练轮数
epochs_baseline = 30
epochs_quantile = 30
epochs_caun_stage2 = 30

# batch size和学习率
batch_size = 64
learning_rate = 0.001

# CAUN特定参数
decoder_dim = 100
num_heads = 5
num_decoder_layers = 2
```

## 输出结果

- `results/all_methods_metrics.csv` - 所有方法的评估指标
- `results/all_methods_predictions.csv` - 所有方法的预测结果
- `results/*.png` - 可视化图表
- `checkpoint/*.pt` - 训练好的模型

## 项目结构

```
1031/
├── main.py                    # 主训练脚本
├── config.py                  # 配置文件
├── data_utils.py              # 数据处理工具
├── metrics.py                 # 评估指标
├── visualize.py               # 可视化工具
├── models/                    # 模型模块
│   ├── baseline_model.py
│   ├── quantile_regression.py
│   ├── gp_approximation.py
│   ├── npkde.py
│   └── adaptive_caun.py       # 我们的方法
├── power.xlsx                 # 数据文件
├── checkpoint/                # 模型保存目录
└── results/                   # 结果输出目录
```

## GPU支持

代码自动检测并使用GPU（CUDA/MPS），如无GPU则使用CPU。
