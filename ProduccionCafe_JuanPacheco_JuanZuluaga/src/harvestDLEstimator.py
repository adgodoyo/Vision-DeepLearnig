import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
warnings.filterwarnings('ignore')

class HarvestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.dropna(subset=['days_to_harvest'])
        self.data = self.data[self.data['days_to_harvest'] > 0] 
        self.img_dir = img_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.data['class_encoded'] = self.label_encoder.fit_transform(self.data['class'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['class'], self.data.iloc[idx]['filename'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (48, 48), color=(128, 128, 128)) # imagen dummy

        if self.transform:
            image = self.transform(image)
                        
        class_label = self.data.iloc[idx]['class_encoded']
        days_to_harvest = float(self.data.iloc[idx]['days_to_harvest'])
        
        return image, torch.tensor([class_label], dtype=torch.float32), torch.tensor([days_to_harvest], dtype=torch.float32)

class HarvestCNN(nn.Module):
    """CNN architecture"""
    
    def __init__(self, dropout_rate=0.3, num_filters=32):
        super(HarvestCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.25),
            
            # Block 2
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            # Block 3
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 4 * 16 + 1, 256),  # +1 for class feature
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
    def forward(self, image, class_feature):
        x = self.features(image)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, class_feature], dim=1)
        x = self.classifier(x)
        return x

class CNNTrainer:
    """hyperparameter optimization"""
    
    def __init__(self, csv_file, img_dir):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_model = None
        self.best_params = {}
        self.train_losses = []
        self.val_losses = []
        
    def create_data_loaders(self, batch_size=16):
        """Create optimized data loaders with augmentation"""
        
        train_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        full_dataset = HarvestDataset(self.csv_file, self.img_dir, transform=val_transform)
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(0.70 * total_size)
        val_size = int(0.25 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_dataset.dataset.transform = train_transform
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion, scheduler=None):
        """Train for one epoch with mixed precision"""
        model.train()
        total_loss = 0.0
        scaler = torch.cuda.amp.GradScaler()
        
        for batch_idx, (images, classes, targets) in enumerate(train_loader):
            images, classes, targets = images.to(self.device), classes.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images, classes)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if scheduler:
                scheduler.step()
        
        return total_loss / len(train_loader)
    
    def validate(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0.0
        predictions = []
        actual = []
        
        with torch.no_grad():
            for images, classes, targets in val_loader:
                images, classes, targets = images.to(self.device), classes.to(self.device), targets.to(self.device)
                
                outputs = model(images, classes)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy().flatten())
                actual.extend(targets.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(val_loader)
        r2 = r2_score(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        
        return avg_loss, r2, mae, predictions, actual
    
    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""
        
        params = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.6),
            'num_filters': trial.suggest_categorical('num_filters', [16, 32, 64]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'step', 'plateau'])
        }
        
        train_loader, val_loader, _ = self.create_data_loaders(params['batch_size'])
        
        model = HarvestCNN(dropout_rate=params['dropout_rate'], num_filters=params['num_filters'])
        model.to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = nn.MSELoss()
        
        if params['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        elif params['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        

        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(50):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, 
                                        scheduler if params['scheduler'] != 'plateau' else None)
            val_loss, val_r2, val_mae, _, _ = self.validate(model, val_loader, criterion)
            
            if params['scheduler'] == 'plateau':
                scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                break
            
            # Report to Optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return best_val_loss
    
    def optimize_hyperparameters(self, n_trials=20):
        """Optimize hyperparameters using Optuna"""
        print("Starting CNN hyperparameter optimization...")
        
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        print(f"Best CNN parameters: {self.best_params}")
        return self.best_params
    
    def train_best_model(self, epochs=100):
        print("Training best CNN model...")
        
        train_loader, val_loader, test_loader = self.create_data_loaders(self.best_params['batch_size'])
        
        model = HarvestCNN(
            dropout_rate=self.best_params['dropout_rate'],
            num_filters=self.best_params['num_filters']
        )
        model.to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), 
                              lr=self.best_params['lr'], 
                              weight_decay=self.best_params['weight_decay'])
        criterion = nn.MSELoss()
        
        if self.best_params['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif self.best_params['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion,
                                        scheduler if self.best_params['scheduler'] != 'plateau' else None)
            val_loss, val_r2, val_mae, _, _ = self.validate(model, val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if self.best_params['scheduler'] == 'plateau':
                scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}')
            
            if patience_counter >= max_patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        # Load best model and evaluate on test set
        model.load_state_dict(self.best_model)
        test_loss, test_r2, test_mae, test_preds, test_actual = self.validate(model, test_loader, criterion)
        
        print(f"\nResultados finales:")
        print(f"Test Loss (MSE): {test_loss:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test RMSE: {np.sqrt(test_loss):.4f}")
        
        return model, test_loss, test_r2, test_mae