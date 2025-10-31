from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt


class CausalResize(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.padding_size = padding_size

    def forward(self, x):
        return x[..., : - self.padding_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, n_features, n_filters, filter_size, dilation=1, dropout_rate=0.1):
        super().__init__()
        self.padding_size = filter_size - 1
        # in_channels: 输入特征维度, out_channels: 输出通道数，卷积核数量
        # self.conv = weight_norm(nn.Conv1d(in_channels=n_features, out_channels=n_filters, kernel_size=filter_size,
        #                                  stride=1, padding=self.padding_size, dilation=dilation))
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_filters, kernel_size=filter_size,
                              stride=1, padding=self.padding_size, dilation=dilation)
        self.resize = CausalResize(padding_size=self.padding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        # self.conv_ = weight_norm(nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=filter_size,
        #                                   stride=1, padding=self.padding_size, dilation=dilation))
        self.conv_ = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=filter_size,
                               stride=1, padding=self.padding_size, dilation=dilation)
        self.resize_ = CausalResize(padding_size=self.padding_size)
        self.relu_ = nn.ReLU()
        self.dropout_ = nn.Dropout(p=dropout_rate)

        self.net = nn.Sequential(self.conv, self.resize, self.relu, self.dropout)
        #                         self.conv_, self.resize_, self.relu_, self.dropout_)

        self.conv_residual = nn.Conv1d(in_channels=n_features, out_channels=n_filters, kernel_size=1) \
            if n_features != n_filters else None
        self.relu__ = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x_ = self.net(x)
        residual = x if self.conv_residual is None else self.conv_residual(x)
        return self.relu__(x_ + residual).permute(0, 2, 1)


class TCN(nn.Module):
    def __init__(self, n_features, n_filters, n_timesteps, filter_size):
        super().__init__()
        self.tcn = TCNBlock(n_features=n_features, n_filters=n_filters, filter_size=filter_size)
        self.fc = nn.Linear(in_features=n_timesteps * n_filters, out_features=length_size)

    def forward(self, x):
        o = self.tcn(x)
        o = torch.reshape(o, shape=(o.size(0), -1))  # Flatten
        o = self.fc(o)
        o = o.squeeze()
        return o


def data_loader(window, length_size, batch_size, data):
    # 构建lstm输入
    seq_len = window  # 模型每次输入序列输入序列长度
    sequence_length = seq_len + length_size  # 序列长度，也就是输入序列的长度+预测序列的长度
    result = []  # 空列表
    for index in range(len(data) - sequence_length):  # 循环次数为数据集的总长度
        result.append(data[index: index + sequence_length])  # 第i行到i+sequence_length
    result = np.array(result)  # 得到样本，样本形式为sequence_length*特征
    x_train = result[:, :-length_size]  # 训练集特征数据
    print('x_train shape:', x_train.shape)
    y_train = result[:, -length_size:, -1]  # 训练集目标数据
    print('y_train shape:', y_train.shape)
    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], data_dim))  # 重塑数据形状，保证数据顺利输入模型
    y_train = np.reshape(y_train, (y_train.shape[0], -1))

    X_train, y_train = torch.tensor(X_train).to(torch.float32), torch.tensor(y_train).to(
        torch.float32)  # 将数据转变为tensor张量
    ds = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,shuffle=True)  # 对训练集数据进行打包，每32个数据进行打包一次，组后一组不足32的自动打包
    return dataloader, X_train, y_train



def model_train():
    net = TCN(n_features=n_feature, n_filters=n_filter, n_timesteps=window, filter_size=2)
    criterion = nn.MSELoss()  # 损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 优化算法和学习率
    """
    模型训练过程
    """
    iteration = 0
    for epoch in range(epochs):  # 10
        for i, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()
            if length_size == 1:
                preds = net(datapoints).unsqueeze(1)
            else:
                preds = net(datapoints)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 100 == 0:  # 250
                print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss))
    best_model_path = 'checkpoint/best_TCN.pt'
    torch.save(net.state_dict(), best_model_path)
    return net


def model_test():
    net = TCN(n_features=n_feature, n_filters=n_filter, n_timesteps=window, filter_size=2)
    net.load_state_dict(torch.load('checkpoint/best_TCN.pt'))  # 加载训练好的模型
    net.eval()
    if length_size == 1:
        pred = net(X_test).unsqueeze(1)
    else:
        pred = net(X_test)
    pred = pred.detach().cpu()
    true = y_test.detach().cpu()
    
    # 使用已经拟合好的scaler进行反变换
    pred_uninverse = scaler.inverse_transform(pred)
    true_uninverse = scaler.inverse_transform(true)

    return true_uninverse, pred_uninverse
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

if __name__ == "__main__":
    window = 40  # 模型输入序列长度
    length_size = 1  # 预测结果的序列长度
    epochs = 100  # 迭代次数
    batch_size = 32
    
    # 读取数据
    data = pd.read_excel('power1.xlsx')
    data = data.iloc[:, 1:]  # 第一列为时间，去除时间列
    data_target = data.iloc[:, -1:]  # 目标数据
    data_dim = len(data.columns)  # 修改：使用columns获取特征维度
    data_length = len(data)
    train_set = 0.995

    # 首先划分训练集和测试集
    train_idx = int(train_set * data_length)
    data_train = data.iloc[:train_idx]  # 修改：使用iloc进行切片
    data_test = data.iloc[train_idx:]  # 修改：使用iloc进行切片
    
    # 创建并仅使用训练集拟合scaler
    scaler = preprocessing.MinMaxScaler()
    if data_dim == 1:
        # 只使用训练数据拟合scaler
        scaler.fit(np.array(data_train).reshape(-1, 1))
        # 分别转换训练集和测试集
        data_train_scaled = scaler.transform(np.array(data_train).reshape(-1, 1))
        data_test_scaled = scaler.transform(np.array(data_test).reshape(-1, 1))
    else:
        # 只使用训练数据拟合scaler
        scaler.fit(np.array(data_train))
        # 分别转换训练集和测试集
        data_train_scaled = scaler.transform(np.array(data_train))
        data_test_scaled = scaler.transform(np.array(data_test))

    n_feature = data_dim  # 输入特征维数
    n_filter = 8  # TCN通道个数

    dataloader_train, X_train, y_train = data_loader(window, length_size, batch_size, data_train_scaled)
    dataloader_test, X_test, y_test = data_loader(window, length_size, batch_size, data_test_scaled)

    model_train()
    true, pred = model_test()

    # 保存预测结果到CSV文件
    result_finally = np.concatenate((true, pred), axis=1)
    print(result_finally.shape)
    time = np.arange(len(result_finally))
    
    # 创建DataFrame并保存到CSV
    test_results_df = pd.DataFrame({
        'Timestamp': time,
        'True_Value': result_finally[:, 0].flatten(),
        'Predicted_Value': result_finally[:, 1].flatten()
    })
    test_results_df.to_csv('TCN_test_results.csv', index=False)
    print("测试结果已保存到 TCN_test_results.csv")
    
    # 可视化结果
    plt.figure(figsize=(12, 3))
    plt.plot(time, result_finally[:, 0], c='red', linestyle='--', linewidth=3, label='true')
    plt.plot(time, result_finally[:, 1], c='black', linestyle='-', linewidth=3, label='pred')
    plt.title('TCN Quantile prediction results')
    plt.legend()
    plt.savefig('TCN_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 计算评估指标
    y_test = result_finally[:, 0]
    y_test_predict = result_finally[:, 1]
    R2 = 1 - np.sum((y_test - y_test_predict) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    MAE = mean_absolute_error(y_test_predict, y_test)
    RMSE = np.sqrt(mean_squared_error(y_test_predict, y_test))
    MAPE = mape(y_test_predict, y_test)
    
    # 保存评估指标到CSV
    metrics_df = pd.DataFrame({
        'Metric': ['R2', 'MAE', 'RMSE', 'MAPE'],
        'Value': [R2, MAE, RMSE, MAPE]
    })
    metrics_df.to_csv('TCN_metrics.csv', index=False)
    print("评估指标已保存到 TCN_metrics.csv")

    # 打印评估指标
    print('MAE:', MAE)
    print('RMSE:', RMSE)
    print('MAPE:', MAPE)
    print('r2:', R2)
