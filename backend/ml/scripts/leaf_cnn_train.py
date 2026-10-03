import os
import sys
import time
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------
# 1. Dataset Definition
# ---------------------------------------------------------
class LeafDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        
        csv_path = os.path.join(folder_path, '_classes.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Classes CSV not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        self.filename_col = df.columns[0]
        self.class_cols = df.columns[1:].tolist()
        
        self.samples = []
        for _, row in df.iterrows():
            fname = str(row[self.filename_col]).strip()
            img_path = os.path.join(folder_path, fname)
            if os.path.exists(img_path):
                one_hot = row[self.class_cols].values.astype(np.float32)
                label = int(np.argmax(one_hot))
                self.samples.append((img_path, label))
            else:
                print(f"Warning: Image file missing: {img_path}", flush=True)
                
        print(f"Loaded {len(self.samples)} valid samples from {folder_path}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# ---------------------------------------------------------
# 2. Model Creation
# ---------------------------------------------------------
def create_model(num_classes, model_name='efficientnet_b0'):
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

# ---------------------------------------------------------
# 3. Training & Validation Functions
# ---------------------------------------------------------
def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if scaler is not None and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    total = len(all_labels)
    eval_loss = running_loss / total
    eval_acc = accuracy_score(all_labels, all_preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return eval_loss, eval_acc, precision, recall, f1, all_labels, all_preds

# ---------------------------------------------------------
# 4. Main Training Pipeline
# ---------------------------------------------------------
def main():
    dataset_dir = os.path.abspath(os.path.dirname(__file__))
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')
    
    print(f"Dataset Root: {dataset_dir}", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB", flush=True)
    
    # Image Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Datasets
    train_dataset = LeafDataset(train_dir, transform=train_transform)
    valid_dataset = LeafDataset(valid_dir, transform=val_transform)
    test_dataset = LeafDataset(test_dir, transform=val_transform)
    
    class_names = train_dataset.class_cols
    num_classes = len(class_names)
    print(f"\nClasses ({num_classes}): {class_names}\n", flush=True)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
    
    # Initialize Model
    model = create_model(num_classes, model_name='efficientnet_b0').to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    epochs = 25
    best_valid_f1 = 0.0
    best_model_path = os.path.join(dataset_dir, 'best_leaf_disease_model.pth')
    
    print("=" * 70, flush=True)
    print("STARTING TRAINING ON GPU" if device.type == 'cuda' else "STARTING TRAINING ON CPU", flush=True)
    print("=" * 70, flush=True)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, valid_loader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{epochs:02d}] LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}%", flush=True)
        
        if val_f1 > best_valid_f1:
            best_valid_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_names': class_names
            }, best_model_path)
            print(f"  >>> Best Model Saved! (Val F1: {val_f1*100:.2f}%, Val Acc: {val_acc*100:.2f}%)", flush=True)
            
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.2f} minutes.", flush=True)
    
    # ---------------------------------------------------------
    # 5. Final Testing and Verification
    # ---------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("EVALUATING BEST MODEL ON TEST SET", flush=True)
    print("=" * 70, flush=True)
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    
    print(f"\nFinal Test Set Results:", flush=True)
    print(f"Test Accuracy  : {test_acc*100:.2f}%", flush=True)
    print(f"Test Precision : {test_prec*100:.2f}%", flush=True)
    print(f"Test Recall    : {test_rec*100:.2f}%", flush=True)
    print(f"Test F1-Score  : {test_f1*100:.2f}%", flush=True)
    
    print("\nDetailed Classification Report:", flush=True)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4), flush=True)
    
    print("Confusion Matrix:", flush=True)
    cm = confusion_matrix(y_true, y_pred)
    print(cm, flush=True)
    
    if test_acc >= 0.80 and test_f1 >= 0.80:
        print("\n SUCCESS: Performance requirement (>80% accuracy & F1 score) met!", flush=True)
    else:
        print("\n WARNING: Performance metrics below target threshold of 80%.", flush=True)

if __name__ == '__main__':
    main()
