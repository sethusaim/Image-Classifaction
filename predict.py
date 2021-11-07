import json
import torch

from src.model import get_model
from utils.image_utils import transform_image

model = get_model()

model.load_state_dict(
    torch.load("checkpoints/model.pth", map_location="cpu"), strict=False
)

model.eval()

class_idx = json.load(open("utils/annotations/annot.json"))


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)

    outputs = model.forward(tensor)

    _, y_hat = outputs.max(1)

    predicted_idx = str(y_hat.item())

    return class_idx[predicted_idx]
