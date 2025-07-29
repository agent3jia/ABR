import xml.etree.ElementTree as ET
import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt


class ABRAnalyzer:
    """ABR 数据处理与分析类库，支持多种XML格式和V型波形检测"""

    def __init__(self):
        self.time: np.ndarray = np.array([])
        self.voltage: np.ndarray = np.array([])
        self.sample_rate: float = 0.0

    def load_from_xml(self, xml_content: str, format_type: str = "hearlab") -> None:
        """
        从XML内容加载ABR数据（支持两种格式）

        :param xml_content: XML字符串
        :param format_type: "hearlab"（PointsValue格式）或 "ep25"（Value列表格式）
        """
        if format_type == "hearlab":
            self._parse_hearlab(xml_content)
        elif format_type == "ep25":
            self._parse_ep25(xml_content)
        else:
            raise ValueError("Unsupported XML format. Choose 'hearlab' or 'ep25'.")

    def _parse_hearlab(self, xml_content: str) -> None:
        """解析第一种XML格式（含PointsValue标签）"""
        root = ET.fromstring(xml_content)
        points_str = root.find(".//PointsValue").text
        points = [tuple(map(float, pt.strip("()").split(","))) for pt in points_str.split("),(")]
        self.time = np.array([pt[0] for pt in points])
        self.voltage = np.array([pt[1] for pt in points])
        self.sample_rate = 1 / (self.time[1] - self.time[0]) * 1000  # 估算采样率（Hz）

    def _parse_ep25(self, xml_content: str) -> None:
        """解析第二种XML格式（含Value列表）"""
        root = ET.fromstring(xml_content)
        waveform = root.find(".//Waveform")
        # self.sample_rate = float(waveform.get("SampleRate"))
        voltage_values = waveform.find(".//IPSI_A_Raw").findall("Value")
        self.voltage = np.array([float(v.text) for v in voltage_values])
        self.time = np.arange(0, len(self.voltage)) / self.sample_rate * 1000  # 转换为毫秒

    def detect_v_waves(self, min_prominence: float = 10.0) -> List[Tuple[float, float]]:
        """
        检测V型波形（波谷+两侧波峰）

        :param min_prominence: 峰值/谷值的最小显著度
        :return: 列表[(时间, 电压)]
        """
        valleys, _ = find_peaks(-self.voltage, prominence=min_prominence)
        peaks, _ = find_peaks(self.voltage, prominence=min_prominence)

        v_waves = []
        for v in valleys:
            left_peak = peaks[peaks < v][-1] if any(peaks < v) else None
            right_peak = peaks[peaks > v][0] if any(peaks > v) else None

            if left_peak and right_peak:
                left_height = self.voltage[left_peak] - self.voltage[v]
                right_height = self.voltage[right_peak] - self.voltage[v]
                if left_height > min_prominence and right_height > min_prominence:
                    v_waves.append((self.time[v], self.voltage[v]))
        return v_waves

    def label_peaks(self) -> Dict[str, Optional[Tuple[float, float]]]:
        """标注ABR关键波（I、III、V）"""
        peaks, _ = find_peaks(self.voltage, prominence=10, distance=int(0.5 * self.sample_rate))
        peaks = sorted(peaks, key=lambda x: self.time[x])

        labels = {}
        if len(peaks) >= 1:
            labels["wave_I"] = (self.time[peaks[0]], self.voltage[peaks[0]])
        if len(peaks) >= 3:
            labels["wave_III"] = (self.time[peaks[1]], self.voltage[peaks[1]])
            labels["wave_V"] = (self.time[peaks[2]], self.voltage[peaks[2]])
        return labels

    def plot(self, v_waves: List[Tuple[float, float]] = None, peaks: Dict[str, Tuple[float, float]] = None) -> None:
        """绘制ABR信号和标注"""
        plt.figure(figsize=(12, 4))
        plt.plot(self.time, self.voltage, label="ABR Signal")

        if v_waves:
            v_times, v_voltages = zip(*v_waves)
            plt.scatter(v_times, v_voltages, color="red", s=100, label="V-Wave")

        if peaks:
            for name, (t, v) in peaks.items():
                plt.scatter(t, v, marker="*", s=200, label=name)

        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (µV)")
        plt.title("ABR Analysis")
        plt.legend()
        plt.grid()
        plt.show()

