import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform =torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.cifar.CIFAR10(root="./dataset",transform=dataset_transform, train = True, download=True)
test_set = torchvision.datasets.cifar.CIFAR10(root="./dataset", transform=dataset_transform,train = True, download=True)

# print(test_set[0])
# print(test_set.classes)

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set",img ,i)
writer.close()