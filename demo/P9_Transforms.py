from PIL import Image
from torchvision import transforms

img_path = "C:\\ABR\\Demo\\data\\train\\ants\\0013035.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# print(tensor_img)

