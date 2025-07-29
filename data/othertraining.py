# # 加载模型
# processor = ABRProcessorTorch.load_model("abr_model.pth")
#
# # 预测单条信号
# test_signal = np.random.randn(800)  # 示例信号
# prob = processor.predict(test_signal, sr=15000)
# print(f"Abnormality Probability: {prob:.3f}")
#
# # 批量预测
# batch_signals = [...]  # 信号列表
# probs = [processor.predict(sig) for sig in batch_signals]


# # 在train方法中添加
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
# # 在每个epoch后调用
# scheduler.step(val_loss)