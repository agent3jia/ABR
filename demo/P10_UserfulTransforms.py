from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer = SummaryWriter("logs")

img = Image.open("C:\\ABR\\Demo\\data\\train\\ants\\0013035.jpg")

trans_tensor =transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image("toTensor",img_tensor)
writer.close()

# Normalize
print(img_tensor[0][0][0])
trans_nor = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_nor(img_tensor)
print(img_norm[0][0][0])

writer.add_image("Normalize",img_norm)
# print(img)