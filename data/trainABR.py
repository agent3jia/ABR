from CNNBiLSTMABR import ABRProcessorTorch
#初始化处理器
processor = ABRProcessorTorch(target_length=500)
#加载数据
X, y = processor.load_dataset(r".\hearlab")

#创建数据加载器
train_loader, val_loader = processor.create_dataloaders(X, y, batch_size=32)

#构建并训练模型
processor.build_model()
processor.train(train_loader, val_loader, epochs=30)

# 评估
test_loader = ...  # 创建测试集加载器
results = processor.evaluate(test_loader)
print(f"Test Accuracy: {results['accuracy']:.2%}")

# 保存模型
processor.save_model("abr_model.pth")