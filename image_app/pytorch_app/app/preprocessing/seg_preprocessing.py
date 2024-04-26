import torchvision.transforms as v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = {
    'FCN_ResNet50': v2.Compose([
        v2.ToTensor(),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
    'DeepLabV3_ResNet50': v2.Compose([
        v2.ToTensor(),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
    'LRASPP_MobileNet': v2.Compose([
        v2.ToTensor(),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
}