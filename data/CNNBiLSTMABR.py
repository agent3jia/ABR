import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import os
import matplotlib.pyplot as plt


class ABRDataset(Dataset):
    """PyTorch数据加载类"""

    def __init__(self, signals, labels):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class CNNBiLSTMModel(nn.Module):
    """PyTorch版CNN-BiLSTM模型"""

    def __init__(self, input_size=1, hidden_size=64, num_classes=1):
        super(CNNBiLSTMModel, self).__init__()

        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )

        # BiLSTM部分
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入形状: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # 转为(batch, features, seq_len)适应Conv1d

        # CNN处理
        x = self.cnn(x)  # 输出: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # 转为(batch, seq_len, channels)适应LSTM

        # BiLSTM处理
        lstm_out, _ = self.lstm(x)  # 输出: (batch, seq_len, hidden_size*2)

        # 取最后一个时间步
        out = lstm_out[:, -1, :]

        # 分类
        return self.fc(out)


class ABRProcessorTorch:
    """PyTorch版ABR处理管道"""

    def __init__(self, target_length=500, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.target_length = target_length
        self.device = torch.device(device)
        self.model = None
        self.sample_rate = None
        self.classes = {'normal': 0, 'abnormal': 1}
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def load_single_file(self, filepath: str) -> Tuple[np.ndarray, float]:
        """加载单个XML文件"""
        tree = ET.parse(filepath)
        root = tree.getroot()

        if root.find(".//PointsValue") is not None:
            points_str = root.find(".//PointsValue").text
            points = [tuple(map(float, pt.strip("()").split(",")))
                      for pt in points_str.split("),(")]
            time = np.array([pt[0] for pt in points])
            voltage = np.array([pt[1] for pt in points])
            sr = 1 / (time[1] - time[0]) * 1000
        # else:
        #     waveform = root.find(".//Waveform")
        #     voltage = np.array([float(v.text) for v in waveform.find(".//IPSI_A_Raw").findall("Value")])
        #     sr = float(waveform.get("SampleRate"))

        self.sample_rate = sr
        return voltage, sr

    def preprocess(self, voltage: np.ndarray, sr: float) -> np.ndarray:
        """数据预处理"""
        voltage = (voltage - np.mean(voltage)) / np.std(voltage)
        if len(voltage) != self.target_length:
            voltage = resample(voltage, self.target_length)
        return voltage.reshape(-1, 1)  # (seq_len, features)

    def load_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """递归加载完整数据集（包括子目录）"""
        X, y = [], []
        # 递归遍历所有子目录和文件
        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('.xml'):
                    file_path = os.path.join(root, filename)
                    try:
                        voltage, sr = self.load_single_file(file_path)
                        processed = self.preprocess(voltage, sr)
                        X.append(processed)
                        # y.append(class_id)
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")

        return np.array(X), np.array(y)

    def create_dataloaders(self, X: np.ndarray, y: np.ndarray,
                           batch_size=32, test_size=0.2) -> Tuple[DataLoader, DataLoader]:
        """创建PyTorch数据加载器"""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

        train_dataset = ABRDataset(X_train, y_train)
        val_dataset = ABRDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def build_model(self, input_size=1, hidden_size=64) -> nn.Module:
        """构建模型"""
        self.model = CNNBiLSTMModel(input_size, hidden_size).to(self.device)
        return self.model

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs=50, lr=0.001, patience=5):
        """训练模型"""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss, correct, total = 0, 0, 0

            for signals, labels in train_loader:
                signals = signals.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(signals)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            # 验证阶段
            val_loss, val_correct, val_total = self._evaluate(val_loader, criterion)

            # 记录历史
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(correct / total)
            self.history['val_acc'].append(val_correct / val_total)

            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {self.history['train_loss'][-1]:.4f} | "
                  f"Val Loss: {self.history['val_loss'][-1]:.4f} | "
                  f"Train Acc: {self.history['train_acc'][-1]:.2f} | "
                  f"Val Acc: {self.history['val_acc'][-1]:.2f}")

            # Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

    def _evaluate(self, loader: DataLoader, criterion) -> Tuple[float, int, int]:
        """评估辅助函数"""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for signals, labels in loader:
                signals = signals.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                outputs = self.model(signals)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(loader), correct, total

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        criterion = nn.BCELoss()
        loss, correct, total = self._evaluate(test_loader, criterion)
        return {'loss': loss, 'accuracy': correct / total}

    def predict(self, signal: np.ndarray, sr: float = None) -> float:
        """预测单条信号"""
        if sr is None:
            if self.sample_rate is None:
                raise ValueError("Sampling rate must be provided")
            sr = self.sample_rate

        processed = self.preprocess(signal, sr)
        signal_tensor = torch.FloatTensor(processed).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(signal_tensor)

        return output.item()

    def plot_training_history(self, save_path=None):
        """可视化训练过程"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_length': self.target_length,
            'classes': self.classes
        }, path)

    @classmethod
    def load_model(cls, path, device='auto'):
        """加载模型"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(path, map_location=device)
        processor = cls(target_length=checkpoint['target_length'], device=device)
        processor.classes = checkpoint['classes']

        processor.build_model()
        processor.model.load_state_dict(checkpoint['model_state_dict'])
        processor.model.to(processor.device)

        return processor