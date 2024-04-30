import torch
from dataset import VineyardDataset
from torch.utils.data import DataLoader
import torchvision
import time

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    train_transform,
    val_transform,
    batch_size,
    num_workers,
    pin_memory = True
):
    train_data = VineyardDataset(images_dir=train_dir,
                                masks_dir=train_maskdir,
                                transform=train_transform)
    val_data = VineyardDataset(images_dir=val_dir,
                                masks_dir=val_maskdir,
                                transform=val_transform)

    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory)

    return train_dataloader, val_dataloader

def save_checkpoint(state, filename = "/content/drive/MyDrive/VineNetData/saved_images/my_checkpoint.pth.tar"):
    print("==> Saving CheckPoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("==> Loading CheckPoint")
    model.load_state_dict(checkpoint["state_dict"])


# Define the evaluation metrics functions
def calculate_precision(tp, fp):
    if (tp+fp) == 0:
        return 0.0
    return tp / (tp + fp)

def calculate_recall(tp, fn):
    if (tp+fn) == 0:
        return 0.0
    return tp / (tp + fn)

def calculate_mAP(tp, fp, fn):
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    if precision == 0 or recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

def measure_inference_time(model, data_loader):
    model.eval()
    total_time = 0
    with torch.no_grad():
        for images, masks in data_loader:
            if torch.cuda.is_available():
                images = images.cuda()
            start_time = time.time()
            _ = model(images)
            end_time = time.time()
            total_time += end_time - start_time
    return total_time / len(data_loader)

# Assuming test_loader is defined similar to train_loader
def evaluate_model(model, loss_fn, val_loader):
    model.eval()
    tp, fp, fn = 0, 0, 0
    total_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                masks = masks.cuda()

            outputs = model(images)
            masks = masks.reshape(outputs.shape)
            loss = loss_fn(outputs, masks)
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

    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, mAP: {mAP:.4f}, Inference Time: {inference_time:.4f}, Loss: {total_loss / len(val_loader):.4f}')


def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.inference_mode():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(X))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum())/((preds + y).sum()+ 1e-8)
    print(
        f"Got {num_correct}/{num_pixels} with accuracy {(num_correct/num_pixels)*100: .3f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(loader, model,device, folder_dir = "saved_images/"):
    model.eval()

    for idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        with torch.inference_mode():
            preds = torch.sigmoid(model(X))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, f"{folder_dir}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder_dir}{idx}.png")

    model.train()