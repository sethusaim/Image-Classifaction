import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm


def train(data_loader, model, optimizer, device, loss_function):
    try:
        model.train()

        running_loss = 0

        for bidx, (images, targets) in enumerate(tqdm(data_loader)):
            images = images.to(device)

            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_function(outputs, targets)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            targets = targets.detach().cpu().numpy()

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        train_epoch_loss = running_loss / len(data_loader.dataset)

        train_acc = accuracy_score(targets, outputs)

        return train_epoch_loss, train_acc

    except Exception as e:
        print(str(e))


def evaluate(data_loader, model, device, loss_function):
    try:
        model.eval()

        running_loss = 0

        with torch.no_grad():
            for bidx, (images, targets) in enumerate(tqdm(data_loader)):
                images = images.to(device)

                targets = targets.to(device)

                outputs = model(images)

                loss = loss_function(outputs, targets)

                running_loss += loss.item() * images.size(0)

                targets = targets.detach().cpu().numpy()

                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

            val_epoch_loss = running_loss / len(data_loader.dataset)

            val_acc = accuracy_score(targets, outputs)

        return val_epoch_loss, val_acc

    except Exception as e:
        print(str(e))


def get_weighted_random_sampler_for_imagefolder(dataset):
    try:
        class_weights = []

        data_dir = dataset.root

        for root, subdir, files in os.walk(data_dir):
            if len(files) > 0:
                class_weights.append(1 / len(files))

        sample_weights = [0] * len(dataset)

        for idx, (data, label) in enumerate(dataset.imgs):
            class_weight = class_weights[label]

            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        return sampler

    except Exception as e:
        print(str(e))


def get_weighted_random_sampler_for_custom_dataset(dataset):
    try:
        print("Applying weighted random sampler for custom dataset")

        counts = np.bincount(dataset.targets)

        label_weights = 1.0 / counts

        weights = label_weights[dataset.targets]

        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )

        print("Applied weighted random sampler for custom dataset")

        return sampler

    except Exception as e:
        print(str(e))
