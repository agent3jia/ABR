import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn

class ABRAnalyzer:
    """ABR数据分析类，通过时间点自动获取V值"""

    def __init__(self):
        self.time: np.ndarray = np.array([])
        self.voltage: np.ndarray = np.array([])
        self.derivative: np.ndarray = np.array([])
        self.sample_rate: float = 0.0
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    def to_tensor(self):
        return torch.FloatTensor(self.voltage).view(1, 1, -1)  # 形状(1, 1, seq_len)

    def load_from_xml(self, xml_content: str, format_type: str = "hearlab") -> None:
        """
        从XML内容加载ABR数据

        :param xml_content: XML字符串
        :param format_type: 目前仅支持"hearlab"格式
        """
        if format_type == "hearlab":
            self._parse_hearlab(xml_content)
        else:
            raise ValueError("目前仅支持'hearlab'格式")

    def _parse_hearlab(self, xml_content: str) -> None:
        """解析hearlab格式的XML数据"""
        root = ET.fromstring(xml_content)
        points_str = root.find(".//PointsValue").text
        points = [tuple(map(float, pt.strip("()").split(","))) for pt in points_str.split("),(")]
        self.time = np.array([pt[0] for pt in points])
        self.voltage = np.array([pt[1] for pt in points])
        self.derivative = np.gradient(self.voltage)
        # self.sample_rate = 1 / (self.time[1] - self.time[0]) * 1000  # 计算采样率(Hz)

    def set_v_peak_by_time(self, v_time: float) -> None:
        """
        通过时间点设置V峰值（自动获取对应电压值）

        :param v_time: V波的时间(ms)
        """
        if not (self.time[0] <= v_time <= self.time[-1]):
            raise ValueError(f"V时间{v_time}ms不在数据范围内[{self.time[0]}, {self.time[-1]}]")

        # 找到最接近的时间点
        idx = np.argmin(np.abs(self.time - v_time))
        self.v_peak = (self.time[idx], self.voltage[idx])

    def plot(self, highlight_range: Tuple[float, float] = None) -> None:
        """
        绘制ABR信号

        :param highlight_range: 可选，高亮显示的时间范围(start, end)
        """

        plt.figure(figsize=(12, 4))
        plt.plot(self.time, self.voltage, label="ABR信号")

        if hasattr(self, 'v_peak'):
            plt.scatter(self.v_peak[0], self.v_peak[1],
                        color="red", s=100, marker="*",
                        label=f"V峰值({self.v_peak[0]:.2f}ms, {self.v_peak[1]:.2f}µV)")

        if highlight_range:
            plt.axvspan(highlight_range[0], highlight_range[1],
                        color='yellow', alpha=0.2, label='高亮区域')

        plt.xlabel("时间(ms)")
        plt.ylabel("电压(µV)")
        plt.title("ABR分析结果")
        plt.legend()
        plt.grid()
        plt.show()

    def print_derivative(self):
        print("时间点(ms) | 幅值(μV) | 导数(μV/ms)")
        for t, a, d in zip(self.time, self.voltage, self.derivative):
            print(f"{t:.4f} | {a:7.1f} | {d:7.1f}")
        # input_data = np.stack([self.voltage, self.derivative], axis=1)  # Shape: (681, 2)

class CNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN模块
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),  # 输入通道=2（幅值+导数）
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # BiLSTM模块
        self.lstm = nn.LSTM(input_size=64, hidden_size=128,
                           num_layers=2, bidirectional=True, batch_first=True)
        # 输出头
        self.head = nn.Sequential(
            nn.Linear(256, 1),  # 输出每个时间点的V波概率
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)  # [Batch, 64, 100]
        x = x.permute(0, 2, 1)  # → [Batch, 100, 64]
        x, _ = self.lstm(x)  # [Batch, 100, 256]
        return self.head(x)  # [Batch, 100, 1]

class SimpleABRModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN部分：提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),  # 输入1通道，输出16通道
            nn.ReLU(),
            nn.MaxPool1d(2)  # 降采样到340点
        )
        # BiLSTM部分：分析时序关系
        self.lstm = nn.LSTM(16, 32, bidirectional=True, batch_first=True)  # 双向LSTM
        # 输出层
        self.head = nn.Linear(64, 1)  # 64=32*2（双向）
    def forward(self, x):
        # x形状: [batch_size, 1, 681]
        x = self.cnn(x)          # -> [batch_size, 16, 340]
        x = x.permute(0, 2, 1)   # -> [batch_size, 340, 16] (LSTM需要时间步在前)
        x, _ = self.lstm(x)      # -> [batch_size, 340, 64]
        return torch.sigmoid(self.head(x))  # -> [batch_size, 340, 1]

import torch
import torch.nn as nn

class CNNBiLSTM2(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN模块
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),  # 输入通道=2（幅值+导数）
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # BiLSTM模块
        self.lstm = nn.LSTM(input_size=64, hidden_size=128,
                           num_layers=2, bidirectional=True, batch_first=True)
        # 输出头
        self.head = nn.Sequential(
            nn.Linear(256, 1),  # 输出每个时间点的V波概率
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)  # [Batch, 64, 100]
        x = x.permute(0, 2, 1)  # → [Batch, 100, 64]
        x, _ = self.lstm(x)  # [Batch, 100, 256]
        return self.head(x)  # [Batch, 100, 1]

class VWaveDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 双通道CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),  # 输入通道=1
            nn.ReLU(),
            nn.MaxPool1d(2)  # 681 -> 340
        )
        # BiLSTM
        self.lstm = nn.LSTM(16, 32, bidirectional=True)
        # 输出正波概率
        self.head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)  # [1, 16, 340]
        x = x.permute(0, 2, 1)  # [1, 340, 16]
        x, _ = self.lstm(x)  # [1, 340, 64]
        return self.head(x)  # [1, 340, 1]
# 使用示例
if __name__ == "__main__":
    analyzer = ABRAnalyzer()

    # 加载XML文件
    with open(r"C:\ABR\data\hearlab\ABRI14R\Rerun R_4k_75.xml", "r") as f:
        xml_content = f.read()

    analyzer.load_from_xml(xml_content)

    model = VWaveDetector()

    input_tensor = analyzer.to_tensor()

    prob = model(input_tensor)

    print("输出概率范围:", prob.min().item(), "~", prob.max().item())
    plt.plot(prob[0, :, 0].detach().numpy())  # 形状变为 (208,)
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("V波概率预测")
    plt.show()

    # analyzer.print_derivative()
    #
    # # 通过时间点设置V峰值（自动获取对应电压值）
    # analyzer.set_v_peak_by_time(v_time=6.15)  # 只需要指定时间，自动获取电压
    #
    # # 绘制结果，可选高亮显示区域
    # analyzer.plot(highlight_range=(6.0, 7.0))

