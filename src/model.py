import torch.nn as nn
from torchvision.models.resnet import resnet152


def get_model():
    try:
        model = resnet152(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=1024, out_features=120),
        )

        return model

    except Exception as e:
        print(str(e))
