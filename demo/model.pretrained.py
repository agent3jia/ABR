import torchvision
from mpmath.identification import transforms

# train_data = torchvision.datasets.imagenet("./data_Image_net",split='train',download = "True",transforms=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1 )

vgg16_true = torchvision.models.vgg16( )

print("ok")