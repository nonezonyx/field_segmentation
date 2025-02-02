import torchvision.models as models
import torch

def get_model(load_state_path=None):
    model = models.segmentation.deeplabv3_resnet50(weights="DeepLabV3_ResNet50_Weights.DEFAULT")
    model.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1))
    if load_state_path is not None:
        model.load_state_dict(torch.load(load_state_path, map_location=torch.device('cpu')))
    return model