import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import logging
import sys
import os
import argparse
import datetime

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

# train model
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import copy

from utils import get_max_memory_allocated
from model import get_model

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = y_pred.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    return train_loss / len(train_loader), correct / total

def eval_model(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)
            val_loss += loss.item()

            _, predicted = y_pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    return val_loss / len(val_loader), correct / total


def main(args):

    SCRATCH = args.scratch
    CLASSES = args.classes
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr

    date = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    experiment_name = f'{date}-{args.model}-{"scratch" if SCRATCH else "imagenet1k"}'
    experiment_folder = f'exps/{experiment_name}'
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    log_file = f'{experiment_folder}/log.txt'
    # set logging with INFO level and date format 
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] - %(message)s', handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ])

    logging.info(f'Experiment: {experiment_name}')
    logging.info(f'Model: {args.model}')
    logging.info(f'Scratch: {SCRATCH}')
    logging.info(f'Classes: {CLASSES}')
    logging.info(f'Epochs: {EPOCHS}')
    logging.info(f'Batch size: {BATCH_SIZE}')
    logging.info(f'Learning rate: {LR}')


    # image datase
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing()
    ])


    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_data = ImageFolder('/home/je689690/CAP5516/chest_xray/train', transform=train_transform)
    test_data = ImageFolder('/home/je689690/CAP5516/chest_xray/test', transform=val_transform)
    val_data = ImageFolder('/home/je689690/CAP5516/chest_xray/val', transform=val_transform)

    # model
    model = get_model(CLASSES, SCRATCH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=LR)


    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []
    MAX_EARLY_STOPPING = 20
    early_stopping = MAX_EARLY_STOPPING
    best_val_loss = np.inf
    best_model = None

    train_loss, train_acc = eval_model(model, train_loader, criterion, device)
    val_loss, val_acc = eval_model(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    gpu_memory = get_max_memory_allocated()
    logging.info(f'Epoch {0}/{EPOCHS}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}, GPU memory: {gpu_memory:.2f} MB')

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping = MAX_EARLY_STOPPING
            best_model = copy.deepcopy(model)
            logging.info('Best model updated epoch:', epoch)
        else:
            early_stopping -= 1

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        gpu_memory = get_max_memory_allocated()
        logging.info(f'Epoch {epoch + 1}/{EPOCHS}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}, GPU memory: {gpu_memory:.2f} MB')

        if early_stopping == 0:
            logging.info('Early stopping')
            break

    # Save history
    history = pd.DataFrame({
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies
    })

    filename = os.path.join(experiment_folder, f'history.csv')
    model_path = os.path.join(experiment_folder, f'last.pth')
    history.to_csv(filename, index=False)

    # save model
    torch.save(best_model.state_dict(), model_path)


    # evaluate model
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    test_loss, test_acc = eval_model(best_model, test_loader, criterion, device)
    logging.info(f'Test loss: {test_loss:.4f}, test acc: {test_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
