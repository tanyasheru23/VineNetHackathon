import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim
from model import UNet
from torch.utils.data import DataLoader
from dataset import VineyardDataset
import time
import numpy as np

from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
BATCH_SIZE = 16
EPOCHS = 30
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/content/drive/MyDrive/VineNetData/train/images"
TRAIN_MASK_DIR = "/content/drive/MyDrive/VineNetData/train/masks"
VAL_IMG_DIR = "/content/drive/MyDrive/VineNetData/val/images"
VAL_MASK_DIR = "/content/drive/MyDrive/VineNetData/val/masks"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    train_losses = []
    for batch_idx, (data, targets) in enumerate(loop):
        model.to(device=DEVICE)
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)

        # forward
        model.train()
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        train_losses.append(loss.to("cpu").detach().numpy())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())
        return train_losses

### TRAINING
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std = [1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]
)

val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std = [1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]
)

new_model = UNet(in_channels=3, out_channels=1).to(device=DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=new_model.parameters(), lr=LEARNING_RATE)

train_loader, val_loader = get_loaders(train_dir=TRAIN_IMG_DIR,
                                        train_maskdir=TRAIN_MASK_DIR,
                                        val_dir=VAL_IMG_DIR,
                                        val_maskdir=VAL_MASK_DIR,
                                        train_transform=train_transform,
                                        val_transform=val_transform,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY)

if LOAD_MODEL:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), new_model)
scaler = torch.cuda.amp.GradScaler()

train_losses = []
val_losses = []
# loop = tqdm(train_loader)
for epoch in range(EPOCHS):
    print(f"------Epoch: {epoch}------")

    # train_losses = train_fn(train_loader, new_model, optimizer, loss_fn, scaler)
    loop = tqdm(train_loader)

    train_loss = []
    for batch_idx, (data, targets) in enumerate(loop):
        new_model.to(device=DEVICE)
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)

        # forward
        new_model.train()
        predictions = new_model(data)
        loss = loss_fn(predictions, targets)
        train_loss.append(loss.to("cpu").detach().numpy())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())
    train_losses.append(sum(train_loss)/len(train_loss))
    # if(len(train_losses)>3):
    #   if(abs(train_losses[-1]-train_losses[-2])<0.0001):
    #     checkpoint = {"state_dict": new_model.state_dict(),"optimizer": optimizer.state_dict()}
    #     save_checkpoint(checkpoint)
    #     break

    # save the new_model
    checkpoint = {
        "state_dict": new_model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    save_checkpoint(checkpoint)


    check_accuracy(val_loader, new_model, device= DEVICE)
    # evaluate_new_model(new_model, loss_fn, val_loader)

    # print some examples to a folder
    save_predictions_as_imgs(
        val_loader,
        new_model,
        folder_dir="/content/drive/MyDrive/VineNetData/saved_images",
        device=DEVICE
    )

    test_loss = []
    # test_acc = 0
    new_model.eval()
    with torch.inference_mode():
        loop = tqdm(val_loader)
        val_loss = []
        for batch_idx, (data, targets) in enumerate(loop):
            new_model.to(device=DEVICE)
            data = data.to(device = DEVICE)
            targets = targets.float().unsqueeze(1).to(device = DEVICE)

            # forward
            new_model.eval()
            predictions = new_model(data)
            loss = loss_fn(predictions, targets)
            val_loss.append(loss.to("cpu").detach().numpy())

            # loop.set_postfix(loss = loss.item())
        val_losses.append(sum(val_loss)/len(val_loss))

# LOSS VS EPOCH GRAPH
import matplotlib.pyplot as plt

# Plot losses versus epochs
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-')
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='*', linestyle='--')  # Use '--' for the linestyle
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.grid(False)
plt.show()


# Define the evaluation metrics functions
def calculate_precision(tp, fp):
    return tp / (tp + fp)

def calculate_recall(tp, fn):
    return tp / (tp + fn)

def calculate_mAP(tp, fp, fn):
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    return (2 * precision * recall) / (precision + recall)

#------------------------------------

def compute_iou_and_miou(outputs, labels, num_classes):
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    for cls in range(num_classes):
        intersection[cls] = torch.logical_and(outputs == cls, labels == cls).sum()
        union[cls] = torch.logical_or(outputs == cls, labels == cls).sum()

    iou = torch.div(intersection, union)
    valid_classes = union.nonzero().size(0)
    miou = iou.sum() / valid_classes

    return iou, miou

#----------------------------------------


def jaccard_index(gt, pred):
    pred = pred.to("cpu").numpy()
    gt = gt.to("cpu").numpy()
    pred = np.where(pred >= 0.5, 1, 0)
    gt = (gt >= 0.5).astype(int)
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    jaccard = intersection / (union + 1e-8)
    return jaccard

def measure_inference_time(model, data_loader):
    model.eval()
    total_time = 0
    with torch.no_grad():
        for images, masks in val_loader:
            if torch.cuda.is_available():
                images = images.cuda()
            start_time = time.time()
            _ = model(images)
            end_time = time.time()
            total_time += end_time - start_time
    return total_time / len(data_loader)

# Assuming test_loader is defined similar to train_loader
def evaluate_model(model, criterion, val_loader):
    model.eval()
    tp, fp, fn = 0, 0, 0
    total_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                masks = masks.cuda()

            outputs = model(images)
            masks = masks.unsqueeze(1)
            # print(masks.shape)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Calculate true positives, false positives, and false negatives
            predictions = (outputs > 0.5).float()
            tp += torch.sum(predictions * masks).item()
            fp += torch.sum(predictions * (1 - masks)).item()
            fn += torch.sum((1 - predictions) * masks).item()

    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    mAP = calculate_mAP(tp, fp, fn)
    inference_time = measure_inference_time(model, val_loader)
    iou, miou = compute_iou_and_miou(predictions, masks, 1)
    jacc_list = []
    for i in range(len(masks)):
        jacc_list.append((jaccard_index(masks[i], predictions[i])))
    jacc_list = np.array(jacc_list)

    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, mAP: {mAP:.4f}, miou: {miou}, Jaccard_index: {np.mean(jacc_list)}, Inference Time: {inference_time:.4f}, Loss: {total_loss / len(val_loader):.4f}')

evaluate_model(new_model, loss_fn, val_loader)
