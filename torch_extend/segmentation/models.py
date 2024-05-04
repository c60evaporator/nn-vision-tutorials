from typing import Dict
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision import transforms, models
import random
import matplotlib.pyplot as plt
import time
import os

from .display import show_segmentations, show_predicted_segmentation_minibatch
from .metrics import segmentation_ious_torchvison

class BaseSegmentation():

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DEFAULT_RESIZE = (224, 224)

    OPTIMIZER_PARAMS = {'lr': 0.0005}

    def _get_denormalize_transform(self):
        return transforms.Compose([
            transforms.Normalize(mean=[-mean/std for mean, std in zip(self.IMAGENET_MEAN, self.IMAGENET_STD)],
                                std=[1/std for std in self.IMAGENET_STD])
        ])
    
    def __init__(self, train_dataset: VisionDataset, val_dataset: VisionDataset,
                 device: str, batch_size: int, num_load_workers: int, 
                 idx_to_class: Dict[int, str], bg_idx: int = 0, border_idx: int = None,
                 seed: int = None,
                 mlflow_tracking_uri=None, mlflow_artifact_location=None, mlflow_experiment_name=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # Set the device name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            device = 'cpu'
        # Set the random seed
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        # Create data loader
        self.batch_size = batch_size
        self.num_load_workers = num_load_workers
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_load_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_load_workers)
        # Register label information
        self.idx_to_class = idx_to_class
        self.bg_idx = bg_idx
        self.border_idx = border_idx
        self.class_to_idx = {v: k for k, v in idx_to_class.items()}
        self.num_classes = len(idx_to_class) if bg_idx is None else len(idx_to_class) + 1  # Classification classes
        # Register MLFlow settings
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_artifact_location = mlflow_artifact_location
        self.mlflow_experiment_name = mlflow_experiment_name
        
    def display_annotations(self, num_displayed_images: int = 10, batch_idx: int=0, 
                            denormalize_transform=None):
        """Display images in the specified mini-batch"""
        for i, (imgs, targets) in enumerate(self.train_loader):
            if i == batch_idx:
                break
        for i, (img, target) in enumerate(zip(imgs, targets)):
            if denormalize_transform is not None:
                img = denormalize_transform(img)
            img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
            show_segmentations(img, target, self.idx_to_class, bg_idx=self.bg_idx, border_idx=self.border_idx)
            if num_displayed_images is not None and i >= num_displayed_images - 1:
                break

    def load_model(self, model: nn.Sequential):
        self.model = model
        print(model)
        model.to(self.device)

    def train(self, num_epochs: int):
        raise NotImplementedError
    
    def save_trained_weight(self, path: str):
        """Save the trained weight"""
        raise NotImplementedError

    def load_weight(self, path: str):
        raise NotImplementedError
    
    def show_prediction(self, num_displayed_images: int = 10, batch_idx: int=0, result_convert_func=None,
                        denormalize_transform=None):
        """Display the prediction in the specified mini-batch"""
        # Inference
        for i, (imgs, targets) in enumerate(self.val_loader):
            if i == batch_idx:
                break
        self.model.eval()  # Set the evaluation mode
        imgs_gpu = imgs.to(self.device)
        predictions = self.model(imgs_gpu)
        # Result 
        if result_convert_func is not None:
            predictions = result_convert_func(predictions)
        # Reverse normalization for getting the raw image
        if denormalize_transform is not None:
            imgs_display = [self.denormalize_transform(img) for img in imgs]
        else:
            imgs_display = [img for img in imgs]
        # Show the image
        show_predicted_segmentation_minibatch(imgs_display, predictions, targets, self.idx_to_class,
                                              bg_idx=self.bg_idx, plot_raw_image=True,
                                              max_displayed_images=num_displayed_images)

    def mean_iou(self, dataloader):
        raise NotImplementedError
    

class TorchVisionSeg(BaseSegmentation):
    class ModelNameEnum(Enum):
        FCN = 'FCN'
        DeepLabV3 = 'DeepLabV3'
        LRASPP = 'LRASPP'
    class BackboneEnum(Enum):
        ResNet50 = 'ResNet50'
        ResNet101 = 'ResNet101'
        MobileNet = 'MobileNet'

    @classmethod
    def _default_criterion(cls, inputs, target):
        losses = {}
        for name, x in inputs.items():
            losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
        if len(losses) == 1:
            return losses["out"]
        return losses["out"] + 0.5 * losses["aux"]

    def load_model(self, model_name: ModelNameEnum, backbone: BackboneEnum,
                   freeze_pretrained: bool = True, replace_whole_classifier: bool = False):
        ###### Load the model ######
        # FCN        
        if model_name == self.ModelNameEnum.FCN.name:
            if backbone == self.BackboneEnum.ResNet50.name:
                weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT
                model = models.segmentation.fcn_resnet50(weights=weights)
            elif backbone == self.BackboneEnum.ResNet101.name:
                weights = models.segmentation.FCN_ResNet101_Weights.DEFAULT
                model = models.segmentation.fcn_resnet101(weights=weights)
            else:
                raise Exception('The `backbone` argument should be "ResNet50" or "ResNet101"')
        # DeepLabV3        
        elif model_name == self.ModelNameEnum.DeepLabV3.name:
            if backbone == self.BackboneEnum.MobileNet.name:
                weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
            elif backbone == self.BackboneEnum.ResNet50.name:
                weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
                model = models.segmentation.deeplabv3_resnet50(weights=weights)
            elif backbone == self.BackboneEnum.ResNet101.name:
                weights = models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
                model = models.segmentation.deeplabv3_resnet101(weights=weights)
            else:
                raise Exception('The `backbone` argument should be "MobileNet" ,"ResNet50", or "ResNet101"')
        # LRASPP
        elif model_name == self.ModelNameEnum.LRASPP.name:
            if backbone == self.BackboneEnum.MobileNet.name:
                weights = models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
                model = models.segmentation.lraspp_mobilenet_v3_large(weights=weights)
            else:
                raise Exception('The `backbone` argument should be "MobileNet"')
        # Freeze pretrained parameters
        if freeze_pretrained:
            for param in model.parameters():
                param.requires_grad = False
        ###### Replace layers for fine tuning ######
        # FCN        
        if model_name == self.ModelNameEnum.FCN.name:
            if replace_whole_classifier:
                # Replace all layes in the classifier
                model.aux_classifier = models.segmentation.fcn.FCNHead(1024, self.num_classes)
                model.classifier = models.segmentation.fcn.FCNHead(2048, self.num_classes)
            else:
                # Replace the last Conv layer of the classifier and the aux_classifier
                inter_channels_classifier = model.classifier[4].in_channels  # Input channels of the classifier
                inter_channels_aux = model.aux_classifier[4].in_channels  # Input channels of the aux_classifier
                model.classifier[4] = nn.Conv2d(inter_channels_classifier, self.num_classes, 1)  # Last Conv layer of the classifier
                model.aux_classifier[4] = nn.Conv2d(inter_channels_aux, self.num_classes, 1)  # Last Conv layer of the classifier
        # DeepLabV3        
        elif model_name == self.ModelNameEnum.DeepLabV3.name:
            if replace_whole_classifier:
                # Replace all layes in the classifier
                model.aux_classifier = models.segmentation.fcn.FCNHead(1024, self.num_classes)
                model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, self.num_classes)
            else:
                # Replace the last Conv layer of the classifier and the aux_classifier 
                inter_channels_aux = model.aux_classifier[4].in_channels  # Input channels of the aux_classifier
                model.classifier[4] = nn.Conv2d(256, self.num_classes, 1)  # Last Conv layer of the classifier
                model.aux_classifier[4] = nn.Conv2d(inter_channels_aux, self.num_classes, 1)  # Last Conv layer of the classifier
        # LRASPP
        elif model_name == self.ModelNameEnum.LRASPP.name:
            if replace_whole_classifier:
                # Replace all layes in the classifier
                model.aux_classifier = models.segmentation.fcn.FCNHead(1024, self.num_classes)
                model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, self.num_classes)
            else:
                # Replace the last Conv layer of the classifier and the aux_classifier 
                inter_channels_aux = model.aux_classifier[4].in_channels  # Input channels of the aux_classifier
                model.classifier[4] = nn.Conv2d(256, self.num_classes, 1)  # Last Conv layer of the classifier
                model.aux_classifier[4] = nn.Conv2d(inter_channels_aux, self.num_classes, 1)  # Last Conv layer of the classifier
        ###### Register the model ######
        # Choose parameters to be trained
        self.params_trained = [p for p in model.parameters() if p.requires_grad]
        # Register the model
        super().load_model(model)
        ###### MLFlow ######

    def _train_batch(self, imgs: torch.Tensor, targets: torch.Tensor):
        """Validate one batch"""
        # Send images and labels to GPU
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        # Calculate the loss
        output = self.model(imgs)  # Forward (Prediction)
        loss = self.criterion(output, targets)  # Calculate criterion
        # Update parameters
        self.optimizer.zero_grad()  # Initialize gradient
        loss.backward()  # Backpropagation (Calculate gradient)
        self.optimizer.step()  # Update parameters (Based on optimizer algorithm)
        return loss

    def _validate_batch(self, val_imgs: torch.Tensor, val_targets: torch.Tensor):
        """Validate one batch"""
        # Calculate the loss
        val_imgs = val_imgs.to(self.device)
        val_targets = val_targets.to(self.device)
        val_output = self.model(val_imgs)  # Forward (Prediction)
        val_loss = self.criterion(val_output, val_targets)  # Calculate criterion
        return val_loss

    def train(self, num_epochs: int,
              optimizer_params = None, criterion = None):
        """Training"""
        # Create the optimizer
        if optimizer_params is None:
            optimizer_params = {}
        self.optimizer_params = optimizer_params
        self.optimizer = optim.Adam(self.params_trained, **optimizer_params)
        # Create the criterion
        if criterion is None:
            criterion = self._default_criterion
        self.criterion = criterion
        ###### Train ######
        self.model.train()  # Set the training mode
        self.losses = []  # Array for string loss (criterion)
        self.val_losses = []  # Array for validation loss
        self.elaped_times = []  # Elapsed times
        start = time.time()  # For elapsed time
        # Epoch loop
        for epoch in range(num_epochs):
            # Initialize training metrics
            running_loss = 0.0  # Initialize running loss
            # Mini-batch loop
            for i, (imgs, targets) in enumerate(self.train_loader):
                # Training
                loss = self._train_batch(imgs, targets)
                running_loss += loss.item()  # Update running loss
                if i%100 == 0:  # Show progress every 100 times
                    print(f'minibatch index: {i}/{len(self.train_loader)}, elapsed_time: {time.time() - start}')
            # Calculate average of running losses and accs
            running_loss /= len(self.train_loader)
            self.losses.append(running_loss)

            # Calculate validation metrics
            val_running_loss = 0.0  # Initialize validation running loss
            with torch.no_grad():
                for i, (val_imgs, val_targets) in enumerate(self.val_loader):
                    val_loss = self._validate_batch(val_imgs, val_targets)
                    val_running_loss += val_loss.item()  # Update running loss
                    if i%100 == 0:  # Show progress every 100 times
                        print(f'val minibatch index: {i}/{len(self.val_loader)}, elapsed_time: {time.time() - start}')
            val_running_loss /= len(self.val_loader)
            self.val_losses.append(val_running_loss)

            self.elaped_times.append(time.time() - start)
            print(f'epoch: {epoch}, loss: {running_loss}, val_loss: {val_running_loss}, elapsed_time: {self.elaped_times[-1]}')
    
        ###### MLFlow ######

    def plot_train_history(self):
        """Plot the loss history"""
        plt.plot(self.losses, label='Train loss')
        plt.plot(self.val_losses, label='Validation loss')
        plt.title('Loss history')
        plt.legend()
        plt.show()

    def save_trained_weight(self, path: str):
        """Save the trained weight"""
        params = self.model.state_dict()
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(params, path)

    def load_weight(self, path: str):
        params_load = torch.load(path)
        self.model.load_state_dict(params_load)
    
    def mean_iou(self, dataloader):
        ious_all = segmentation_ious_torchvison(dataloader, self.model, self.device, self.idx_to_class)
        return ious_all