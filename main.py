import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import ssl
import os
import torch.serialization
from torch.serialization import safe_globals

# Fix SSL certificate issue for macOS
ssl._create_default_https_context = ssl._create_unverified_context

class PCamDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype('float32')/255.0
        label = self.labels[idx]
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0).transpose(1, 2, 0)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        image = Image.fromarray((image * 255).astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

class CancerClassifier(nn.Module):
    def __init__(self):
        super(CancerClassifier, self).__init__()
        # Use ResNet18 instead of 50
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        
        # Simpler classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        self.resnet.fc = self.classifier

    def forward(self, x):
        return self.resnet(x)

class BalancedBCELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(BalancedBCELoss, self).__init__()
        self.alpha = alpha  # Weight for false positives vs false negatives
        
    def forward(self, pred, target):
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Calculate weights for false positives and false negatives
        false_pos_weight = (target == 0).float() * self.alpha
        false_neg_weight = (target == 1).float() * (1 - self.alpha)
        
        # Combine weights
        weights = false_pos_weight + false_neg_weight
        
        # Apply weights and take mean
        weighted_loss = (bce_loss * weights).mean()
        
        return weighted_loss

def main():
    train_x_path = '/Users/harshithyallampalli/Downloads/Lymph_Node_Project/camelyonpatch_level_2_split_train_x.h5-002'
    train_y_path = '/Users/harshithyallampalli/Downloads/Lymph_Node_Project/drive-download-20250510T015435Z-001/camelyonpatch_level_2_split_train_y.h5'
    test_x_path = '/Users/harshithyallampalli/Downloads/Lymph_Node_Project/drive-download-20250510T015435Z-001/camelyonpatch_level_2_split_test_x.h5'
    test_y_path = '/Users/harshithyallampalli/Downloads/Lymph_Node_Project/drive-download-20250510T015435Z-001/camelyonpatch_level_2_split_test_y.h5'

    # Load the data
    with h5py.File(train_x_path, 'r') as f:
        X_train_full = np.array(f['x'])
    with h5py.File(train_y_path, 'r') as f:
        y_train_full = np.array(f['y'])
    
    # Randomly sample 10000 indices
    np.random.seed(42)  # for reproducibility
    total_samples = len(X_train_full)
    random_indices = np.random.choice(total_samples, size=10000, replace=False)
    
    # Use the random indices to select data
    X_train = X_train_full[random_indices]
    y_train = y_train_full[random_indices]
    
    print("Original dataset size:", total_samples)
    print("Sampled dataset size:", len(X_train))
    print("Image shape:", X_train.shape)
    print("Label shape:", y_train.shape)
    
    # Load the test data
    with h5py.File(test_x_path, 'r') as f:
        X_test = np.array(f['x'])
    with h5py.File(test_y_path, 'r') as f:
        y_test = np.array(f['y'])
    print("Test Image shape:", X_test.shape)
    print("Test Label shape:", y_test.shape)

    # Enhanced data augmentation with stronger augmentation for positive cases
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),  # Increased rotation
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation transform remains simple
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    # Enable automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler()
    
    # Create datasets with balanced sampling
    train_dataset = PCamDataset(X_train, y_train, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Smaller batch size for better generalization
        shuffle=True,  # Use shuffle instead of sampler since we're using weighted loss
        num_workers=2,
        pin_memory=False,
        persistent_workers=True
    )
    
    test_dataset = PCamDataset(X_test, y_test, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and move to device
    model = CancerClassifier().to(device)
    
    # Setup mixed precision training based on device
    if device.type == "mps":
        use_amp = False  # MPS doesn't support AMP yet
        scaler = None
    else:
        use_amp = True
        scaler = torch.amp.GradScaler()
    
    # Calculate statistics and class weights
    total_samples = len(y_train)
    pos_samples = np.sum(y_train)
    neg_samples = total_samples - pos_samples
    
    # Using balanced weights since dataset is balanced
    pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Positive samples: {pos_samples}")
    print(f"Negative samples: {neg_samples}")
    print(f"Using balanced weights")
    
    # Use balanced BCE loss
    criterion = BalancedBCELoss(alpha=0.5)  # Equal weight to false positives and negatives
    
    # Optimizer with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    
    # Cosine annealing scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20,  # Number of epochs
        eta_min=1e-6  # Minimum learning rate
    )

    do_train = False  # Enable training with new settings
    if do_train:
        print("Training new ResNet18 model...")
        num_epochs = 20
        best_f1_score = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            all_train_preds = []
            all_train_labels = []
            
            model.train()
            optimizer.zero_grad()
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device).view(-1, 1)
                
                if use_amp:
                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                running_loss += loss.item()
                
                with torch.no_grad():
                    predicted = (torch.sigmoid(outputs) > 0.5).float()  # Balanced threshold
                    all_train_preds.extend(predicted.cpu().numpy())
                    all_train_labels.extend(labels.cpu().numpy())
            
            # Calculate training metrics
            train_loss = running_loss / len(train_loader)
            train_preds = np.array(all_train_preds)
            train_labels = np.array(all_train_labels)
            train_acc = 100 * (np.sum(train_preds == train_labels) / len(train_preds))
            
            # Calculate class-wise accuracy
            train_pos_acc = 100 * np.sum((train_preds == 1) & (train_labels == 1)) / np.sum(train_labels == 1)
            train_neg_acc = 100 * np.sum((train_preds == 0) & (train_labels == 0)) / np.sum(train_labels == 0)
            
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Train Positive Acc: {train_pos_acc:.2f}%, Train Negative Acc: {train_neg_acc:.2f}%")
            
            # Validation
            model.eval()
            val_loss = 0.0
            all_val_preds = []
            all_val_labels = []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device).view(-1, 1)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    predicted = (torch.sigmoid(outputs) > 0.5).float()  # Balanced threshold
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
            
            val_loss = val_loss / len(test_loader)
            val_preds = np.array(all_val_preds)
            val_labels = np.array(all_val_labels)
            val_acc = 100 * (np.sum(val_preds == val_labels) / len(val_preds))
            
            # Calculate class-wise validation accuracy
            val_pos_acc = 100 * np.sum((val_preds == 1) & (val_labels == 1)) / np.sum(val_labels == 1)
            val_neg_acc = 100 * np.sum((val_preds == 0) & (val_labels == 0)) / np.sum(val_labels == 0)
            
            # Calculate F1 score
            tp = np.sum((val_preds == 1) & (val_labels == 1))
            fp = np.sum((val_preds == 1) & (val_labels == 0))
            fn = np.sum((val_preds == 0) & (val_labels == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Val Positive Acc: {val_pos_acc:.2f}%, Val Negative Acc: {val_neg_acc:.2f}%")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"F1 Score: {f1_score:.4f}")
            
            # Save model if F1 score improves
            if epoch == 0 or f1_score > best_f1_score:
                best_f1_score = f1_score
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_pos_acc': val_pos_acc,
                    'val_neg_acc': val_neg_acc,
                    'f1_score': f1_score,
                    'precision': precision,
                    'recall': recall
                }, 'cancer_classifier_bestv2.pth')
                print(f"Saved best model with F1 score: {f1_score:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered! No improvement in F1 score for {patience} epochs.")
                    break
            
            scheduler.step()
            print("-" * 50)
    else:
        model_loaded = False
        checkpoint = torch.load('cancer_classifier_best.pth', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best model checkpoint.")
        model_loaded = True
        if model_loaded:
            model.eval()
            correct = 0
            total = 0
            
            # Initialize confusion matrix
            confusion_matrix = torch.zeros(2, 2)  # 2x2 for binary classification
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device).view(-1, 1)
                    outputs = model(images)
                    predicted = (outputs > 0.5).float()
                    
                    # Update confusion matrix
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[int(t.item()), int(p.item())] += 1
                    
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            # Calculate metrics
            tn = confusion_matrix[0, 0].item()
            fp = confusion_matrix[0, 1].item()
            fn = confusion_matrix[1, 0].item()
            tp = confusion_matrix[1, 1].item()
            
            accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print("\nConfusion Matrix:")
            print("                 Predicted")
            print("                Neg     Pos")
            print(f"Actual Neg |  {tn:6.0f}  {fp:6.0f}")
            print(f"Actual Pos |  {fn:6.0f}  {tp:6.0f}")
            print("\nMetrics:")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"\nDetailed Analysis:")
            print(f"True Negatives: {tn:.0f}")
            print(f"False Positives: {fp:.0f}")
            print(f"False Negatives: {fn:.0f}")
            print(f"True Positives: {tp:.0f}")
            
            try:
                print("\nVisualizing predictions...")
                # Get a random batch of images
                test_iter = iter(test_loader)
                batch_idx = torch.randint(0, len(test_loader), (1,)).item()
                
                # Skip to the random batch
                for _ in range(batch_idx):
                    next(test_iter)
                
                images, labels = next(test_iter)
                
                # Randomly select 20 images from this batch if batch size is larger than 20
                if images.size(0) > 20:
                    indices = torch.randperm(images.size(0))[:20]
                    images = images[indices]
                    labels = labels[indices]
                
                images = images.to(device)
                labels = labels.to(device).view(-1, 1)
                with torch.no_grad():
                    outputs = model(images)
                    preds = (outputs > 0.5).float()

                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                images = images * std + mean

                images_np = images.cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()

                # Calculate number of rows and columns needed for 20 images
                num_images = min(20, len(images_np))
                num_cols = 5
                num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division
                
                plt.figure(figsize=(20, 16))  # Increased figure size for better visibility
                for i in range(num_images):
                    plt.subplot(num_rows, num_cols, i+1)
                    plt.imshow(images_np[i])
                    plt.axis('off')
                    color = 'green' if preds[i].item() == labels[i].item() else 'red'
                    plt.title(f"Pred: {int(preds[i].item())}\nTrue: {int(labels[i].item())}", 
                             color=color, fontsize=10)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error during visualization: {str(e)}")
                print("Continuing with the rest of the program...")

    def evaluate_model(model, data_loader, threshold=0.25):  # Lowered threshold to favor positive predictions
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device).view(-1, 1)
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > threshold).float()  # Lower threshold
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        
        return preds, labels

if __name__ == '__main__':
    main()