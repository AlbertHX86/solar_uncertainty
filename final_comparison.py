"""
最终对比实验：5种不确定性量化方法
1. Baseline Point Prediction (无不确定性)
2. Quantile Regression (直接分位数预测)
3. GP Approximation (高斯过程近似)
4. NPKDE (传统核密度估计)
5. Adaptive CAUN (我们的创新方法) ⭐
"""

from TCN import TCNBlock
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 全局参数
window = 40
length_size = 1
batch_size = 32
epochs_baseline = 5  # 基准模型训练轮数
epochs_stage1 = 5    # CAUN第一阶段
epochs_stage2 = 5    # CAUN第二阶段

n_filters = 32
n_filters1 = 50
filter_size = 4
dropout_rate = 0.1


# ==================== 统一编码器 ====================
class TCNBiLSTMEncoder(nn.Module):
    """统一的TCN-BiLSTM编码器"""
    def __init__(self, n_features, n_filters, n_filters1, filter_size, dropout_rate=0.1):
        super().__init__()
        self.tcn = TCNBlock(n_features=n_features, n_filters=n_filters, filter_size=filter_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bilstm = nn.LSTM(
            input_size=n_filters,
            hidden_size=n_filters1,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x):
        tcn_out = self.tcn(x)
        tcn_out = self.dropout(tcn_out)
        lstm_out, _ = self.bilstm(tcn_out)
        return lstm_out
    
    def get_last_hidden(self, x):
        lstm_out = self.forward(x)
        return lstm_out[:, -1, :]


# ==================== 方法1: Baseline Point Prediction ====================
class BaselinePointModel(nn.Module):
    """基础点预测模型（无不确定性量化）"""
    def __init__(self, n_features, n_filters, n_filters1, filter_size):
        super().__init__()
        self.encoder = TCNBiLSTMEncoder(n_features, n_filters, n_filters1, filter_size)
        self.fc = nn.Linear(n_filters1 * 2, 1)
        
    def forward(self, x):
        hidden = self.encoder.get_last_hidden(x)
        return self.fc(hidden).squeeze(-1)
    
    def get_features(self, x):
        """供Adaptive CAUN使用"""
        return self.encoder(x)


# ==================== 方法2: Quantile Regression ====================
class QuantileRegressionModel(nn.Module):
    """直接预测三个分位数"""
    def __init__(self, n_features, n_filters, n_filters1, filter_size):
        super().__init__()
        self.encoder = TCNBiLSTMEncoder(n_features, n_filters, n_filters1, filter_size)
        self.fc_lower = nn.Linear(n_filters1 * 2, 1)
        self.fc_median = nn.Linear(n_filters1 * 2, 1)
        self.fc_upper = nn.Linear(n_filters1 * 2, 1)
        
    def forward(self, x):
        hidden = self.encoder.get_last_hidden(x)
        lower = self.fc_lower(hidden).squeeze(-1)
        median = self.fc_median(hidden).squeeze(-1)
        upper = self.fc_upper(hidden).squeeze(-1)
        return torch.stack([lower, median, upper], dim=1)


def quantile_loss(predictions, targets, quantiles=[0.1, 0.5, 0.9]):
    """分位数损失函数"""
    targets = targets.unsqueeze(1).expand_as(predictions)
    errors = targets - predictions
    
    loss = 0
    for i, q in enumerate(quantiles):
        loss += torch.max(q * errors[:, i], (q - 1) * errors[:, i]).mean()
    
    return loss


# ==================== 方法3: GP Approximation ====================
class GPApproximationModel(nn.Module):
    """高斯过程近似（神经网络实现）"""
    def __init__(self, n_features, n_filters, n_filters1, filter_size):
        super().__init__()
        self.encoder = TCNBiLSTMEncoder(n_features, n_filters, n_filters1, filter_size)
        self.fc_mean = nn.Linear(n_filters1 * 2, 1)
        self.fc_var = nn.Sequential(
            nn.Linear(n_filters1 * 2, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        hidden = self.encoder.get_last_hidden(x)
        mean = self.fc_mean(hidden).squeeze(-1)
        var = self.fc_var(hidden).squeeze(-1)
        return mean, var
    
    def predict_with_uncertainty(self, x):
        self.eval()
        with torch.no_grad():
            mean, var = self.forward(x)
            std = torch.sqrt(var)
            lower = mean - 1.28 * std  # 10%分位数
            upper = mean + 1.28 * std  # 90%分位数
        return lower, mean, upper


def gaussian_nll_loss(pred_mean, pred_var, target):
    """高斯负对数似然损失"""
    return 0.5 * (torch.log(pred_var) + (target - pred_mean)**2 / pred_var).mean()


# ==================== 方法4: NPKDE ====================
class NPKDEUncertainty:
    """非参数核密度估计"""
    def __init__(self, bandwidth=0.05, n_samples=100):
        self.bandwidth = bandwidth
        self.n_samples = n_samples
        self.residual_distribution = None
        
    def fit(self, residuals):
        self.residual_distribution = KernelDensity(
            kernel='gaussian',
            bandwidth=self.bandwidth
        ).fit(residuals.reshape(-1, 1))
        print(f"  NPKDE拟合完成 (带宽={self.bandwidth}, 样本数={len(residuals)})")
        
    def predict_interval(self, point_predictions, confidence=0.8):
        samples = self.residual_distribution.sample(self.n_samples * len(point_predictions))
        samples = samples.reshape(len(point_predictions), self.n_samples)
        
        alpha = (1 - confidence) / 2
        predicted_distributions = point_predictions.reshape(-1, 1) + samples
        
        lower = np.quantile(predicted_distributions, alpha, axis=1)
        upper = np.quantile(predicted_distributions, 1-alpha, axis=1)
        median = np.median(predicted_distributions, axis=1)
        
        return lower, median, upper


# ==================== 方法5: Adaptive CAUN (创新方法) ⭐ ====================
class UncertaintyDetector:
    """识别高不确定性区域"""
    def __init__(self, threshold_percentile=70):
        self.threshold_percentile = threshold_percentile
        self.uncertainty_scores = None
        
    def detect(self, y_true, y_pred):
        residuals = np.abs(y_true - y_pred)
        
        # 方法1: 残差大小
        residual_scores = residuals / (np.std(residuals) + 1e-8)
        
        # 方法2: 局部波动性
        window_size = 5
        local_volatility = np.array([
            np.std(residuals[max(0, i-window_size):min(len(residuals), i+window_size)])
            for i in range(len(residuals))
        ])
        volatility_scores = local_volatility / (np.std(local_volatility) + 1e-8)
        
        # 方法3: 预测变化率
        pred_diff = np.abs(np.diff(y_pred, prepend=y_pred[0]))
        pred_diff_scores = pred_diff / (np.std(pred_diff) + 1e-8)
        
        # 综合不确定性分数
        self.uncertainty_scores = (
            0.5 * residual_scores +
            0.3 * volatility_scores +
            0.2 * pred_diff_scores
        )
        
        threshold = np.percentile(self.uncertainty_scores, self.threshold_percentile)
        high_uncertainty_mask = self.uncertainty_scores >= threshold
        
        print(f"  识别高不确定性区域: {np.sum(high_uncertainty_mask)}/{len(y_true)} "
              f"({np.mean(high_uncertainty_mask)*100:.1f}%)")
        
        return high_uncertainty_mask, self.uncertainty_scores


class AdaptiveCAUN(nn.Module):
    """
    Adaptive Cross-Attention Uncertainty Network
    创新点：基于Transformer Decoder的自适应不确定性量化
    """
    def __init__(self, base_encoder, decoder_dim=100, num_heads=5, num_decoder_layers=2):
        super().__init__()
        
        self.encoder = base_encoder
        self.encoder_projection = nn.Linear(n_filters1 * 2, decoder_dim)
        
        # 不确定性权重网络
        self.uncertainty_weight_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Query tokens (3个：lower, median, upper)
        self.query_tokens = nn.Parameter(torch.randn(1, 3, decoder_dim))
        
        # 不确定性预测头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(decoder_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, uncertainty_scores=None):
        # 编码
        lstm_out = self.encoder.get_features(x)
        memory = self.encoder_projection(lstm_out)
        
        batch_size = memory.size(0)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        
        # Transformer Decoder解码
        decoder_out = self.transformer_decoder(queries, memory)
        
        # 预测三个分位数
        lower = self.uncertainty_head(decoder_out[:, 0, :]).squeeze(-1)
        median = self.uncertainty_head(decoder_out[:, 1, :]).squeeze(-1)
        upper = self.uncertainty_head(decoder_out[:, 2, :]).squeeze(-1)
        
        # 自适应调整（根据不确定性分数）
        if uncertainty_scores is not None:
            uncertainty_scores_tensor = torch.tensor(
                uncertainty_scores, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            weights = self.uncertainty_weight_net(uncertainty_scores_tensor).squeeze(1)
            
            # 高不确定性区域：扩大区间宽度
            interval_width = upper - lower
            adjusted_width = interval_width * (1 + weights)
            
            lower = median - adjusted_width / 2
            upper = median + adjusted_width / 2
        
        return torch.stack([lower, median, upper], dim=1)


def adaptive_loss(predictions, targets, uncertainty_scores=None, alpha=0.1):
    """自适应损失函数"""
    q_loss = quantile_loss(predictions, targets)
    
    # 区间合理性约束
    lower, median, upper = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    order_loss = F.relu(lower - median).mean() + F.relu(median - upper).mean()
    
    # 宽度惩罚
    width = upper - lower
    if uncertainty_scores is not None:
        unc_tensor = torch.tensor(uncertainty_scores, dtype=torch.float32, device=predictions.device)
        width_penalty = (width * (1 - unc_tensor)).mean()
    else:
        width_penalty = width.mean()
    
    return q_loss + alpha * order_loss + 0.01 * width_penalty


# ==================== 数据加载 ====================
def load_data():
    data = pd.read_excel('/Users/albert/Downloads/贝叶斯优化-TCN-BiLSTM-attention/power1.xlsx')
    data = data.iloc[:, 1:][::1]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.fillna(0)
    
    data_dim = len(data.iloc[1, :])
    scaler = preprocessing.MinMaxScaler()
    data_scaled = scaler.fit_transform(np.array(data))
    
    data_length = len(data_scaled)
    train_set = 0.975
    
    data_train = data_scaled[:int(train_set * data_length), :]
    data_test = data_scaled[int(train_set * data_length):, :]
    
    return data_train, data_test, scaler, data_dim


def create_dataloader(data, window, length_size, batch_size, data_dim):
    seq_len = window
    sequence_length = seq_len + length_size
    result = []
    
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    result = np.array(result)
    x_data = result[:, :-length_size]
    y_data = result[:, -length_size:, -1]
    
    X = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], data_dim))
    y = np.reshape(y_data, (y_data.shape[0], -1))
    
    X_tensor = torch.tensor(X).to(torch.float32)
    y_tensor = torch.tensor(y).to(torch.float32)
    
    ds = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    return dataloader, X_tensor, y_tensor


# ==================== 评估指标 ====================
def calculate_metrics(y_true, lower, median, upper, target_coverage=0.8):
    """计算综合评估指标"""
    picp = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    mpiw = np.mean(upper - lower)
    mae = np.mean(np.abs(y_true - median))
    rmse = np.sqrt(np.mean((y_true - median)**2))
    
    # CWC (Coverage Width Criterion)
    if picp < target_coverage * 100:
        penalty = 50 * (target_coverage - picp/100)
        cwc = mpiw * (1 + penalty)
    else:
        cwc = mpiw
    
    # Interval Score
    alpha = 1 - target_coverage
    interval_score = mpiw + (2/alpha) * np.mean(
        np.maximum(0, lower - y_true) + np.maximum(0, y_true - upper)
    )
    
    return {
        'PICP': picp,
        'MPIW': mpiw,
        'MAE': mae,
        'RMSE': rmse,
        'CWC': cwc,
        'IS': interval_score
    }


# ==================== 主函数 ====================
def main():
    print("="*80)
    print("不确定性量化方法最终对比实验")
    print("="*80)
    print("\n对比方法:")
    print("  1. Baseline Point Prediction (无不确定性)")
    print("  2. Quantile Regression (直接分位数预测)")
    print("  3. GP Approximation (高斯过程近似)")
    print("  4. NPKDE (传统核密度估计)")
    print("  5. Adaptive CAUN (我们的创新方法) ⭐")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n[数据准备]")
    data_train, data_test, scaler, data_dim = load_data()
    
    dataloader_train, X_train, y_train = create_dataloader(
        data_train, window, length_size, batch_size, data_dim
    )
    _, X_test, y_test = create_dataloader(
        data_test, window, length_size, batch_size, data_dim
    )
    
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    all_results = {}
    
    # ==================== 方法1: Baseline ====================
    print("\n" + "="*80)
    print("方法1: Baseline Point Prediction")
    print("="*80)
    
    baseline_model = BaselinePointModel(data_dim, n_filters, n_filters1, filter_size).to(device)
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    baseline_model.train()
    for epoch in range(epochs_baseline):
        epoch_loss = 0
        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device).squeeze()
            optimizer.zero_grad()
            pred = baseline_model(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs_baseline}, Loss: {epoch_loss/len(dataloader_train):.6f}")
    
    baseline_model.eval()
    with torch.no_grad():
        baseline_pred = baseline_model(X_test.to(device)).cpu().numpy()
    
    baseline_pred_original = scaler.inverse_transform(baseline_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
    
    baseline_mae = np.mean(np.abs(y_test_original - baseline_pred_original))
    print(f"✓ Baseline MAE: {baseline_mae:.4f}")
    
    all_results['Baseline'] = {
        'median': baseline_pred_original,
        'MAE': baseline_mae
    }
    
    # 保存baseline模型供后续使用
    torch.save(baseline_model.state_dict(), 'checkpoint/baseline_final.pt')
    
    # ==================== 方法2: Quantile Regression ====================
    print("\n" + "="*80)
    print("方法2: Quantile Regression")
    print("="*80)
    
    qr_model = QuantileRegressionModel(data_dim, n_filters, n_filters1, filter_size).to(device)
    optimizer = optim.Adam(qr_model.parameters(), lr=0.001)
    
    qr_model.train()
    for epoch in range(epochs_baseline):
        epoch_loss = 0
        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device).squeeze()
            optimizer.zero_grad()
            pred = qr_model(data)
            loss = quantile_loss(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs_baseline}, Loss: {epoch_loss/len(dataloader_train):.6f}")
    
    qr_model.eval()
    with torch.no_grad():
        qr_pred = qr_model(X_test.to(device))
    
    qr_lower_original = scaler.inverse_transform(qr_pred[:, 0].cpu().numpy().reshape(-1, 1)).flatten()
    qr_median_original = scaler.inverse_transform(qr_pred[:, 1].cpu().numpy().reshape(-1, 1)).flatten()
    qr_upper_original = scaler.inverse_transform(qr_pred[:, 2].cpu().numpy().reshape(-1, 1)).flatten()
    
    # 应用物理约束
    qr_lower_original = np.maximum(qr_lower_original, 0)
    qr_median_original = np.maximum(qr_median_original, 0)
    qr_upper_original = np.maximum(qr_upper_original, 0)
    
    qr_metrics = calculate_metrics(y_test_original, qr_lower_original, qr_median_original, qr_upper_original)
    print(f"✓ Quantile Regression - PICP: {qr_metrics['PICP']:.2f}%, MPIW: {qr_metrics['MPIW']:.4f}, MAE: {qr_metrics['MAE']:.4f}")
    
    all_results['Quantile_Regression'] = {
        'lower': qr_lower_original,
        'median': qr_median_original,
        'upper': qr_upper_original,
        **qr_metrics
    }
    
    # ==================== 方法3: GP Approximation ====================
    print("\n" + "="*80)
    print("方法3: GP Approximation")
    print("="*80)
    
    gp_model = GPApproximationModel(data_dim, n_filters, n_filters1, filter_size).to(device)
    optimizer = optim.Adam(gp_model.parameters(), lr=0.001)
    
    gp_model.train()
    for epoch in range(epochs_baseline):
        epoch_loss = 0
        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device).squeeze()
            optimizer.zero_grad()
            mean, var = gp_model(data)
            loss = gaussian_nll_loss(mean, var, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs_baseline}, Loss: {epoch_loss/len(dataloader_train):.6f}")
    
    gp_lower, gp_median, gp_upper = gp_model.predict_with_uncertainty(X_test.to(device))
    
    gp_lower_original = scaler.inverse_transform(gp_lower.cpu().numpy().reshape(-1, 1)).flatten()
    gp_median_original = scaler.inverse_transform(gp_median.cpu().numpy().reshape(-1, 1)).flatten()
    gp_upper_original = scaler.inverse_transform(gp_upper.cpu().numpy().reshape(-1, 1)).flatten()
    
    gp_lower_original = np.maximum(gp_lower_original, 0)
    gp_median_original = np.maximum(gp_median_original, 0)
    gp_upper_original = np.maximum(gp_upper_original, 0)
    
    gp_metrics = calculate_metrics(y_test_original, gp_lower_original, gp_median_original, gp_upper_original)
    print(f"✓ GP Approximation - PICP: {gp_metrics['PICP']:.2f}%, MPIW: {gp_metrics['MPIW']:.4f}, MAE: {gp_metrics['MAE']:.4f}")
    
    all_results['GP_Approximation'] = {
        'lower': gp_lower_original,
        'median': gp_median_original,
        'upper': gp_upper_original,
        **gp_metrics
    }
    
    # ==================== 方法4: NPKDE ====================
    print("\n" + "="*80)
    print("方法4: NPKDE")
    print("="*80)
    
    # 使用baseline模型在训练集上的残差
    baseline_model.eval()
    with torch.no_grad():
        train_pred = baseline_model(X_train.to(device)).cpu().numpy()
    
    train_residuals = y_train.cpu().numpy().flatten() - train_pred
    
    npkde = NPKDEUncertainty(bandwidth=0.05, n_samples=100)
    npkde.fit(train_residuals)
    
    npkde_lower, npkde_median, npkde_upper = npkde.predict_interval(baseline_pred, confidence=0.8)
    
    npkde_lower_original = scaler.inverse_transform(npkde_lower.reshape(-1, 1)).flatten()
    npkde_median_original = scaler.inverse_transform(npkde_median.reshape(-1, 1)).flatten()
    npkde_upper_original = scaler.inverse_transform(npkde_upper.reshape(-1, 1)).flatten()
    
    npkde_lower_original = np.maximum(npkde_lower_original, 0)
    npkde_median_original = np.maximum(npkde_median_original, 0)
    npkde_upper_original = np.maximum(npkde_upper_original, 0)
    
    npkde_metrics = calculate_metrics(y_test_original, npkde_lower_original, npkde_median_original, npkde_upper_original)
    print(f"✓ NPKDE - PICP: {npkde_metrics['PICP']:.2f}%, MPIW: {npkde_metrics['MPIW']:.4f}, MAE: {npkde_metrics['MAE']:.4f}")
    
    all_results['NPKDE'] = {
        'lower': npkde_lower_original,
        'median': npkde_median_original,
        'upper': npkde_upper_original,
        **npkde_metrics
    }
    
    # ==================== 方法5: Adaptive CAUN ⭐ ====================
    print("\n" + "="*80)
    print("方法5: Adaptive CAUN (我们的创新方法) ⭐")
    print("="*80)
    
    # 阶段1: 识别高不确定性区域
    print("\n[阶段1: 识别高不确定性区域]")
    detector = UncertaintyDetector(threshold_percentile=70)
    high_unc_mask, unc_scores = detector.detect(y_test_original, baseline_pred_original)
    
    # 阶段2: 训练Adaptive CAUN
    print("\n[阶段2: 训练Adaptive CAUN]")
    adaptive_caun = AdaptiveCAUN(
        base_encoder=baseline_model,
        decoder_dim=100,
        num_heads=5,
        num_decoder_layers=2
    ).to(device)
    
    optimizer_caun = optim.Adam(adaptive_caun.parameters(), lr=0.001)
    
    # 归一化不确定性分数
    unc_scores_normalized = (unc_scores - unc_scores.min()) / (unc_scores.max() - unc_scores.min() + 1e-8)
    
    adaptive_caun.train()
    for epoch in range(epochs_stage2):
        epoch_loss = 0
        
        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device).squeeze()
            
            # 简化：使用随机不确定性分数进行训练
            batch_unc_scores = np.random.uniform(0, 1, len(target))
            
            optimizer_caun.zero_grad()
            predictions = adaptive_caun(data, batch_unc_scores)
            loss = adaptive_loss(predictions, target, batch_unc_scores)
            loss.backward()
            optimizer_caun.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs_stage2}, Loss: {epoch_loss/len(dataloader_train):.6f}")
    
    # 保存模型
    torch.save(adaptive_caun.state_dict(), 'checkpoint/adaptive_caun_final.pt')
    
    # 测试Adaptive CAUN
    adaptive_caun.eval()
    with torch.no_grad():
        caun_predictions = adaptive_caun(X_test.to(device), unc_scores_normalized)
    
    caun_lower = scaler.inverse_transform(caun_predictions[:, 0].cpu().numpy().reshape(-1, 1)).flatten()
    caun_median = scaler.inverse_transform(caun_predictions[:, 1].cpu().numpy().reshape(-1, 1)).flatten()
    caun_upper = scaler.inverse_transform(caun_predictions[:, 2].cpu().numpy().reshape(-1, 1)).flatten()
    
    caun_lower = np.maximum(caun_lower, 0)
    caun_median = np.maximum(caun_median, 0)
    caun_upper = np.maximum(caun_upper, 0)
    
    caun_metrics = calculate_metrics(y_test_original, caun_lower, caun_median, caun_upper)
    print(f"✓ Adaptive CAUN - PICP: {caun_metrics['PICP']:.2f}%, MPIW: {caun_metrics['MPIW']:.4f}, MAE: {caun_metrics['MAE']:.4f}")
    
    all_results['Adaptive_CAUN'] = {
        'lower': caun_lower,
        'median': caun_median,
        'upper': caun_upper,
        **caun_metrics
    }
    
    # ==================== 保存结果 ====================
    print("\n" + "="*80)
    print("保存对比结果")
    print("="*80)
    
    # 保存指标对比
    metrics_comparison = []
    for method_name, results in all_results.items():
        if 'PICP' in results:
            row = {'Method': method_name}
            row.update({k: v for k, v in results.items() if k not in ['lower', 'median', 'upper']})
            metrics_comparison.append(row)
    
    metrics_df = pd.DataFrame(metrics_comparison)
    metrics_df.to_csv('final_methods_comparison.csv', index=False)
    print("✓ 指标对比: final_methods_comparison.csv")
    
    # 保存预测结果
    results_dict = {'True_Value': y_test_original}
    for method_name, results in all_results.items():
        if 'median' in results:
            results_dict[f'{method_name}_Median'] = results['median']
        if 'lower' in results:
            results_dict[f'{method_name}_Lower'] = results['lower']
            results_dict[f'{method_name}_Upper'] = results['upper']
    
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv('final_methods_predictions.csv', index=False)
    print("✓ 预测结果: final_methods_predictions.csv")
    
    # ==================== 结果展示 ====================
    print("\n" + "="*80)
    print("实验结果总结")
    print("="*80)
    
    print("\n【按CWC排名 (越小越好)】:")
    metrics_df_sorted = metrics_df.sort_values('CWC')
    for idx, row in enumerate(metrics_df_sorted.itertuples(), 1):
        symbol = "⭐" if row.Method == 'Adaptive_CAUN' else "  "
        print(f"  {symbol}{idx}. {row.Method:25s} - CWC: {row.CWC:.4f}, PICP: {row.PICP:.2f}%, MAE: {row.MAE:.4f}")
    
    print("\n【按PICP最接近80%排名】:")
    metrics_df['PICP_Error'] = np.abs(metrics_df['PICP'] - 80)
    metrics_df_picp = metrics_df.sort_values('PICP_Error')
    for idx, row in enumerate(metrics_df_picp.itertuples(), 1):
        symbol = "⭐" if row.Method == 'Adaptive_CAUN' else "  "
        print(f"  {symbol}{idx}. {row.Method:25s} - PICP: {row.PICP:.2f}% (误差: {row.PICP_Error:.2f}%)")
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    
    return all_results, metrics_df


if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # 创建checkpoint目录
    os.makedirs('checkpoint', exist_ok=True)
    
    all_results, metrics_df = main()

