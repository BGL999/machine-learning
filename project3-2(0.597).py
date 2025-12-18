import os
import json
import warnings
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from tqdm.auto import tqdm
import logging

def setup_logger():
    os.makedirs('./logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("./logs/train.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()
warnings.filterwarnings('ignore')

@dataclass
class TrainConfig:
    train_dir: str = '/kaggle/input/project3/fer_data/fer_data/train'
    test_dir: str = '/kaggle/input/project3/fer_data/fer_data/test'
    model_save_path: str = './models/best_emotion_gray_model.pth'
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 2e-4
    weight_decay: float = 5e-4
    patience: int = 8
    val_split: float = 0.15

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=TrainConfig.train_dir)
    parser.add_argument("--test_dir", type=str, default=TrainConfig.test_dir)
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Ignore unknown args: {unknown}")
    return TrainConfig(**vars(args))

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

set_seed(42)

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("GPU not found, use CPU (very slow)")
    return device

device = setup_device()

EMOTION_MAPPING = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Sad", 4: "Surprise", 5: "Neutral"}
FOLDER_TO_LABEL = {"Angry":0, "Fear":1, "Happy":2, "Sad":3, "Surprise":4, "Neutral":5}

class EmotionGrayDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, is_train: bool = True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        if is_train:
            emotion_folders = [f for f in self.root_dir.glob('*') if f.is_dir() and f.name in FOLDER_TO_LABEL]
            if not emotion_folders:
                logger.error(f"No class folders found! Check path: {root_dir}")
                return
            
            for folder in emotion_folders:
                label = FOLDER_TO_LABEL[folder.name]
                img_paths = list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + \
                            list(folder.glob('*.JPG')) + list(folder.glob('*.PNG'))
                
                self.image_paths.extend(img_paths)
                self.labels.extend([label] * len(img_paths))
        else:
            img_paths = list(self.root_dir.glob('*.jpg')) + list(self.root_dir.glob('*.png')) + \
                        list(self.root_dir.glob('*.JPG')) + list(self.root_dir.glob('*.PNG'))
            if not img_paths:
                logger.error(f"No images in test path! Check path: {root_dir}")
                return
            
            self.image_paths = img_paths
            self.labels = [-1] * len(img_paths)
        
        if len(self.image_paths) == 0:
            logger.error("Dataset is empty! Check path or image format")
        else:
            logger.info(f"Loaded: {len(self.image_paths)} images | train={is_train}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image, label, img_path.name
        except Exception as e:
            logger.warning(f"Load image failed {img_path}: {str(e)[:30]}, return default data")
            return torch.zeros(1, 48, 48), 0, "error_id"

class DataAugmentation:
    @staticmethod
    def get_train_transform():
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    
    @staticmethod
    def get_test_transform():
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

class DataLoaderManager:
    def __init__(self, train_dir: str, batch_size: int = 64, val_split: float = 0.15):
        self.train_transform = DataAugmentation.get_train_transform()
        self.test_transform = DataAugmentation.get_test_transform()
        
        full_dataset = EmotionGrayDataset(train_dir, transform=self.train_transform)
        if len(full_dataset) == 0:
            raise ValueError("Dataset is empty, cannot continue training")
        
        train_idx, val_idx = train_test_split(
            range(len(full_dataset)),
            test_size=val_split,
            stratify=full_dataset.labels,
            random_state=42
        )
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        self.val_dataset.dataset.transform = self.test_transform
        
        self.num_workers = os.cpu_count() // 2 if torch.cuda.is_available() else 0
        self._create_dataloaders(batch_size)

    def _create_dataloaders(self, batch_size: int):
        loader_kwargs = {
            "batch_size": batch_size,
            "pin_memory": torch.cuda.is_available(),
        }
        
        if self.num_workers > 0:
            loader_kwargs["num_workers"] = self.num_workers
            loader_kwargs["persistent_workers"] = True
        
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, **loader_kwargs)
        
        val_loader_kwargs = loader_kwargs.copy()
        val_loader_kwargs["batch_size"] = batch_size * 2
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **val_loader_kwargs)
        
        logger.info(f"Train loader: {len(self.train_loader)} batches | Val loader: {len(self.val_loader)} batches")

class LightEmotionCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        
        with torch.no_grad():
            self.feat_dim = self.features(torch.randn(1,1,48,48)).numel()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

def create_model(num_classes=6):
    model = LightEmotionCNN(num_classes=num_classes).to(device)
    if torch.cuda.is_available() and torch.__version__ >= "2.0.0" and torch.cuda.device_count() == 1:
        model = torch.compile(model)
        logger.info("Enable torch.compile to speed up training")
    
    total_params = sum(p.numel() for p in model.parameters())/10000
    logger.info(f"Model params: {total_params:.1f}k")
    return model

class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs=3, max_epochs=50):
        self.warmup = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            return [base_lr * (self.last_epoch+1)/self.warmup for base_lr in self.base_lrs]
        else:
            decay = 1 - (self.last_epoch-self.warmup)/(self.max_epochs-self.warmup)
            return [base_lr * decay for base_lr in self.base_lrs]

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = WarmupLR(self.optimizer, warmup_epochs=3, max_epochs=config.epochs)
        
        self.scaler = GradScaler()
        self.best_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix({"loss": loss.item(), "acc": correct/total})
        
        return total_loss/total, correct/total*100

    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += images.size(0)
        
        val_loss = total_loss/total
        val_acc = correct/total*100
        logger.info(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.2f}%")
        return val_loss, val_acc

    def train(self):
        os.makedirs('./models', exist_ok=True)
        logger.info("Start training")
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "best_acc": self.best_acc,
                    "config": self.config.__dict__
                }, self.config.model_save_path)
                logger.info(f"Save best model (acc {val_acc:.2f}%) to {self.config.model_save_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stop! Best val acc: {self.best_acc:.2f}%")
                    break
        
        logger.info(f"Training finished! Best acc: {self.best_acc:.2f}%")

class Predictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.transform = DataAugmentation.get_test_transform()
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model = LightEmotionCNN(num_classes=6).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"Model loaded: {model_path} | Best acc: {checkpoint['best_acc']:.2f}%")

    def predict_batch(self, test_dir: str, output_csv: str = "./results/test_predictions.csv"):
        os.makedirs('./results', exist_ok=True)
        test_dataset = EmotionGrayDataset(test_dir, transform=self.transform, is_train=False)
        if len(test_dataset) == 0:
            logger.error("Test dataset is empty!")
            return
        
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0, pin_memory=True)
        results = []
        
        with torch.no_grad():
            for images, _, img_ids in tqdm(test_loader, desc="Prediction"):
                images = images.to(device)
                outputs = self.model(images)
                preds = outputs.argmax(1).cpu().numpy()
                
                for img_id, pred in zip(img_ids, preds):
                    results.append({"ID": img_id, "Emotion": pred})
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        logger.info(f"Prediction CSV saved: {output_csv} | Total {len(df)} records")
        return df

def main():
    config = parse_args()
    
    logger.info("="*50 + "\nLoad dataset")
    try:
        data_manager = DataLoaderManager(config.train_dir, batch_size=config.batch_size, val_split=config.val_split)
    except Exception as e:
        logger.error(f"Data load failed: {str(e)}")
        return
    
    logger.info("\nInitialize model")
    model = create_model(num_classes=6)
    trainer = Trainer(model, data_manager.train_loader, data_manager.val_loader, config)
    trainer.train()
    
    logger.info("\nStart prediction")
    if os.path.exists(config.model_save_path):
        predictor = Predictor(config.model_save_path)
        predictor.predict_batch(config.test_dir)
    else:
        logger.error("Model file not found, skip prediction")
    
    logger.info("\nAll process finished! Results in ./results")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program interrupted manually")
    except Exception as e:
        logger.error(f"Program error: {str(e)}", exc_info=True)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()