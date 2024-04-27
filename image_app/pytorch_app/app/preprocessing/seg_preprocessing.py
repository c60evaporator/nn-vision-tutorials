import torchvision.transforms as v2
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def replace_tensor_value_(tensor, src_border, border_idx):
    tensor[tensor == src_border] = border_idx
    return tensor

def get_transform(model_name):
    transforms = {
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
    return transforms[model_name]
    
def get_target_transform(model_name, border_idx, src_border=255):
    target_transforms = {
        'FCN_ResNet50': v2.Compose([
            v2.PILToTensor(),
            v2.Lambda(lambda x: replace_tensor_value_(x.squeeze(0).long(), src_border, border_idx))
        ]),
        'DeepLabV3_ResNet50': v2.Compose([
            v2.ToTensor(),
            v2.Lambda(lambda x: replace_tensor_value_(x.squeeze(0).long(), src_border, border_idx))
        ]),
        'LRASPP_MobileNet': v2.Compose([
            v2.ToTensor(),
            v2.Lambda(lambda x: replace_tensor_value_(x.squeeze(0).long(), src_border, border_idx))
        ]),
    }
    return target_transforms[model_name]

def get_dataloader(model_name, dataset, batch_size, num_workers):
    dataloaders = {
        'FCN_ResNet50': DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'DeepLabV3_ResNet50': DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'LRASPP_MobileNet': DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return dataloaders[model_name]