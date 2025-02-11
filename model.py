from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from collections import OrderedDict

def get_model_backbone(model):
    # keep module names
    modules = OrderedDict()
    for name, module in list(model.named_children())[:-1]:
        modules[name] = module
    return nn.Sequential(modules)


def get_model(classes, scratch=False):
    weights = ResNet18_Weights.IMAGENET1K_V1 if not scratch else None
    model_type = resnet18
    backbone = get_model_backbone(model_type(weights=weights))

    model_dict = OrderedDict()
    model_dict['backbone'] = backbone
    model_dict['flatten'] = nn.Flatten()
    model_dict['classifier'] = nn.Linear(512, classes)
    model = nn.Sequential(model_dict)

    return model
