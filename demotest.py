import torch


def train(data_loader, model, optimizer, device, loss_function):
    model.train()

    total = 0

    correct = 0

    train_loss = 0

    train_steps = 0

    for i, data in enumerate(data_loader, 0):
        with torch.no_grad():
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            loss = loss_function(outputs, labels)

            train_loss += loss.cpu().numpy()

            train_steps += 1

    train_epoch_loss = train_loss / train_steps

    train_acc = correct / total

    return train_epoch_loss, train_acc


def evaluate(data_loader, model, device, loss_function):
    model.eval()

    val_loss = 0.0

    val_steps = 0

    total = 0

    correct = 0

    for i, data in enumerate(data_loader, 0):
        with torch.no_grad():
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            loss = loss_function(outputs, labels)

            val_loss += loss.cpu().numpy()

            val_steps += 1

        val_epoch_loss = val_loss / val_steps

        val_acc = correct / total

        return val_epoch_loss, val_acc
