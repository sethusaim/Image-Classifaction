import io

from PIL import Image
from torchvision import transforms


def transform_image(image_bytes):
    my_transforms = get_train_aug()

    image = Image.open(io.BytesIO(image_bytes))

    return my_transforms(image).unsqueeze(0)


def get_prediction(model, images_bytes):
    tensor = transform_image(image_bytes=images_bytes)

    output = model.forward(tensor)

    _, y_hat = output.max(1)

    return y_hat


def get_train_aug():
    mean = (0.485, 0.456, 0.406)

    std = (0.229, 0.224, 0.225)

    train_aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=315, scale=(0.95, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_aug


def get_valid_aug():
    mean = (0.485, 0.456, 0.406)

    std = (0.229, 0.224, 0.225)

    valid_aug = transforms.Compose(
        [
            transforms.Resize(size=299),
            transforms.CenterCrop(size=299),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return valid_aug
