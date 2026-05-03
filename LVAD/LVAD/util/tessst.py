import torch

# 加载模型
path1 = "/home/lmy/dateshare/han/SCI/weights/difficult.pt"
path2 = "/home/lmy/dateshare/han/aed-mae/experiments/avenue/checkpoint-best.pth"

model1 = torch.load(path1, map_location="cpu")
model2 = torch.load(path2, map_location="cpu")

checkpoint = torch.load("/home/lmy/dateshare/han/aed-mae/experiments/avenue/checkpoint-best.pth", map_location="cpu")
print("检查点参数键名示例:", list(checkpoint['model'].keys()))