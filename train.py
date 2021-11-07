import os
import warnings

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

from src.dataset import get_imagefolder_dataset
from src.engine import evaluate, get_weighted_random_sampler_for_imagefolder, train
from src.model import get_model
from utils import config
from utils.image_utils import get_train_aug, get_valid_aug

warnings.filterwarnings("ignore")

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device is set to {device}")

    print("Getting the model")

    model = get_model()

    print("Got the model")

    model.to(device=device)

    print(f"Model is on device : {device}")

    best_acc = 0.0

    if not os.path.exists(config.CHECKPOINTS_PATH):
        os.makedirs(config.CHECKPOINTS_PATH)

        print("Checkpoints path created")

    model_save_path = os.path.join(config.CHECKPOINTS_PATH, "model.pth")

    train_aug = get_train_aug()

    print("Got the train augmentations")

    valid_aug = get_valid_aug()

    print("Got the valid augmentations")

    loss_fn = nn.CrossEntropyLoss()

    print("Got the loss function")

    optimizer = Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    print("Got the optimizer")

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=3, threshold=0.9
    )

    print("Got the scheduler")

    train_dataset = get_imagefolder_dataset(
        dir=config.TRAIN_DATA_DIR, transform=train_aug
    )

    print("Got the train dataset")

    print("Applying weighted random sampler for imagefolder dataset")

    train_sampler = get_weighted_random_sampler_for_imagefolder(dataset=train_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        sampler=train_sampler,
    )

    print("Applied weighted random sampler for imagefolder dataset")

    print("Got the train loader")

    valid_dataset = get_imagefolder_dataset(
        dir=config.VAL_DATA_DIR, transform=valid_aug
    )

    print("Got the valid dataset")

    valid_sampler = get_weighted_random_sampler_for_imagefolder(dataset=valid_dataset)

    valid_loader = DataLoader(
        dataset=valid_dataset,
        num_workers=config.NUM_WORKERS,
        batch_size=config.VAL_BATCH_SIZE,
        sampler=valid_sampler,
    )

    print("Got the valid loader")

    print("Started training")

    for epoch in range(1, config.EPOCHS):
        print("==" * 50)

        train_epoch_loss, train_acc = train(
            model=model,
            optimizer=optimizer,
            loss_function=loss_fn,
            data_loader=train_loader,
            device=device,
        )

        print(
            f"Epoch : {epoch} | Train Loss : {train_epoch_loss} | Train Accuracy : {train_acc}"
        )

        valid_epoch_loss, valid_acc = evaluate(
            model=model, data_loader=valid_loader, loss_function=loss_fn, device=device
        )

        scheduler.step(metrics=valid_epoch_loss)

        print(
            f"Epoch : {epoch} | Valid Loss : {valid_epoch_loss} | Valid Accuracy : {valid_acc}"
        )

        if valid_acc > best_acc:
            torch.save(model.state_dict(), model_save_path)

            best_acc = valid_acc

            continue

    print("End of training")

except Exception as e:
    print(str(e))

    raise e
