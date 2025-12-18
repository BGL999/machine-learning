import os
import json
import warnings
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field  # 导入field
import argparse
import random

# 超参数调优依赖
import optuna
from optuna.trial import TrialState
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError  # 导入图片异常类
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
import logging

# ==================== 1. 基础配置 ====================
def setup_logger():
    os.makedirs('./logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("./logs/train_opt.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()
warnings.filterwarnings('ignore')

# 超参数搜索空间
def suggest_hyperparams(trial: optuna.Trial):
    return {
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "lr": trial.suggest_float("lr", 5e-4, 2e-3, log=True),  # 替换deprecated的suggest_loguniform
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.4),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.05, 0.15),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 1.5),
    }

@dataclass
class TrainConfig:
    """修复list默认值+适配鲁棒性"""
    train_dir: str = '/kaggle/input/project3/fer_data/fer_data/train'
    test_dir: str = '/kaggle/input/project3/fer_data/fer_data/test'
    model_save_dir: str = './models'
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    val_split: float = 0.2
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    dropout_rate: float = 0.3
    num_models: int = 3
    seed_list: list = field(default_factory=lambda: [42, 123, 456])  # 修复可变默认值

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=TrainConfig.train_dir)
    parser.add_argument("--test_dir", type=str, default=TrainConfig.test_dir)
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"忽略未知参数: {unknown}")
    return args

# 固定种子
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)} | 显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    else:
        device = torch.device("cpu")
        logger.warning("未检测到GPU，使用CPU训练")
    return device

device = setup_device()

# 类别映射
EMOTION_MAPPING = {0: "愤怒", 1: "恐惧", 2: "快乐", 3: "悲伤", 4: "惊讶", 5: "中性"}
FOLDER_TO_LABEL = {"Angry":0, "Fear":1, "Happy":2, "Sad":3, "Surprise":4, "Neutral":5}
NUM_CLASSES = len(FOLDER_TO_LABEL)

# ==================== 2. 数据处理层（核心修复：图片加载逻辑） ====================
class EmotionGrayDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, is_train: bool = True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        
        # 预加载所有有效图片路径（过滤无效文件）
        self._load_valid_paths()

    def _load_valid_paths(self):
        """预加载并验证图片路径，避免训练时才报错"""
        if self.is_train:
            emotion_folders = [f for f in self.root_dir.glob('*') if f.is_dir() and f.name in FOLDER_TO_LABEL]
            if not emotion_folders:
                logger.error(f"训练集未找到类别目录: {self.root_dir}")
                return
            
            for folder in emotion_folders:
                label = FOLDER_TO_LABEL[folder.name]
                # 遍历所有图片文件
                for img_ext in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
                    for img_path in folder.glob(img_ext):
                        # 预验证图片是否可读
                        try:
                            with Image.open(img_path) as img:
                                img.verify()  # 验证图片完整性
                            self.image_paths.append(img_path)
                            self.labels.append(label)
                        except Exception as e:
                            logger.warning(f"跳过无效训练图片: {img_path} | 错误: {str(e)[:50]}")
        else:
            # 测试集：预加载有效图片
            for img_ext in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
                for img_path in self.root_dir.glob(img_ext):
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        self.image_paths.append(img_path)
                        self.labels.append(-1)
                    except Exception as e:
                        logger.warning(f"跳过无效测试图片: {img_path} | 错误: {str(e)[:50]}")
        
        logger.info(f"加载完成: {len(self.image_paths)} 张有效图片 | 训练集={self.is_train}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """核心修复：鲁棒的图片加载逻辑"""
        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # 显式打开并关闭图片句柄，避免资源泄漏
            with Image.open(img_path) as img:
                # 强制转换为RGB再转灰度，兼容单通道/多通道图片
                img = img.convert('RGB').convert('L')  # 修复不同模式图片的转换问题
                img = img.copy()  # 避免PIL lazy loading导致的异常
            
            # 应用变换
            if self.transform:
                img = self.transform(img)
            
            # 确保返回带后缀的文件名
            img_name = img_path.name if img_path.name.endswith(('.jpg', '.png')) else f"{img_path.stem}.jpg"
            
            return img, label, img_name
        
        except (IOError, UnidentifiedImageError, AttributeError, IndexError) as e:
            # 精准捕获图片相关异常
            logger.warning(f"加载图片失败 {self.image_paths[idx] if idx < len(self.image_paths) else '未知路径'} | 错误: {str(e)[:50]}")
            # 返回兜底数据（确保形状/类型正确）
            return torch.zeros(1, 48, 48, dtype=torch.float32), 0, "error_id.jpg"

# 数据增强（鲁棒版）
class DataAugmentation:
    @staticmethod
    def get_train_transform(dropout_rate=0.3):
        return transforms.Compose([
            transforms.Resize((56, 56)),
            transforms.RandomCrop((48, 48)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            # 调整顺序：先转Tensor，再做RandomErasing
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # 现在处理的是Tensor，有shape属性
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    
    @staticmethod
    def get_test_transform():
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    
    @staticmethod
    def get_tta_transforms():
        """TTA变换（鲁棒版）"""
        return [
            transforms.Compose([transforms.Resize((48, 48)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
            transforms.Compose([transforms.Resize((48, 48)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
            transforms.Compose([transforms.Resize((48, 48)), transforms.ColorJitter(brightness=0.1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
            transforms.Compose([transforms.Resize((48, 48)), transforms.RandomRotation(5), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
            transforms.Compose([transforms.Resize((48, 48)), transforms.ColorJitter(contrast=0.1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
        ]

# 数据加载器（核心修复：关闭多进程）
class DataLoaderManager:
    def __init__(self, train_dir: str, batch_size: int = 64, val_split: float = 0.2, dropout_rate=0.3):
        self.train_transform = DataAugmentation.get_train_transform(dropout_rate)
        self.test_transform = DataAugmentation.get_test_transform()
        
        # 加载数据集（预过滤无效图片）
        train_dataset = EmotionGrayDataset(train_dir, transform=self.train_transform, is_train=True)
        val_dataset = EmotionGrayDataset(train_dir, transform=self.test_transform, is_train=True)
        
        if len(train_dataset) == 0:
            raise ValueError("训练集无有效图片！")
        
        # 分层划分
        train_idx, val_idx = train_test_split(
            range(len(train_dataset)),
            test_size=val_split,
            stratify=train_dataset.labels,
            random_state=42
        )
        self.train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        self.val_dataset = torch.utils.data.Subset(val_dataset, val_idx)
        
        # 类别权重
        train_labels = [train_dataset.labels[i] for i in train_idx]
        class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=train_labels)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        # 加权采样器
        sample_weights = [class_weights[label] for label in train_labels]
        self.sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # 核心修复：num_workers=0 避免多进程图片加载异常
        self.num_workers = 0  # 强制关闭多进程
        self._create_dataloaders(batch_size)

    def _create_dataloaders(self, batch_size: int):
        loader_kwargs = {
            "batch_size": batch_size,
            "pin_memory": torch.cuda.is_available(),
            "num_workers": self.num_workers,  # 0=单进程
            "persistent_workers": False,  # 关闭持久化进程
        }
        
        # 训练加载器
        self.train_loader = DataLoader(self.train_dataset, sampler=self.sampler, **loader_kwargs)
        
        # 验证加载器
        val_loader_kwargs = loader_kwargs.copy()
        val_loader_kwargs["batch_size"] = batch_size * 2
        val_loader_kwargs["shuffle"] = False
        val_loader_kwargs["sampler"] = None
        self.val_loader = DataLoader(self.val_dataset, **val_loader_kwargs)
        
        logger.info(f"训练加载器: {len(self.train_loader)} 批次 | 验证加载器: {len(self.val_loader)} 批次")

# ==================== 3. 模型层（无改动，保持增强版） ====================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EnhancedEmotionCNN(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.features = nn.Sequential(
            ResBlock(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, padding=1),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
            
            ResBlock(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, padding=1),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate + 0.05),
        )
        
        with torch.no_grad():
            self.feat_dim = self.features(self.initial(torch.randn(1,1,48,48))).numel()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate + 0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

def create_model(num_classes=6, dropout_rate=0.3):
    model = EnhancedEmotionCNN(num_classes=num_classes, dropout_rate=dropout_rate).to(device)
    if torch.cuda.is_available() and torch.__version__ >= "2.0.0" and torch.cuda.device_count() == 1:
        try:
            model = torch.compile(model)
            logger.info("启用torch.compile加速训练")
        except Exception as e:
            logger.warning(f"torch.compile失败: {e}")
    total_params = sum(p.numel() for p in model.parameters())/10000
    logger.info(f"模型参数量: {total_params:.1f}万")
    return model

# ==================== 4. 训练器（无核心改动） ====================
class Trainer:
    def __init__(self, model, train_loader, val_loader, config, class_weights, hyperparams):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.class_weights = class_weights
        self.hyperparams = hyperparams
        
        self.criterion = CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=hyperparams["label_smoothing"]
        )
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
            betas=(0.9, 0.999)
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.scaler = GradScaler()
        self.best_acc = 0.0
        self.patience_counter = 0
        self.grad_clip = hyperparams["grad_clip"]

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}", colour='blue')
        
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix({"loss": loss.item(), "acc": correct/total})
        
        self.scheduler.step()
        return total_loss/total, correct/total*100

    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="验证",colour='cyan'):
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += images.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        logger.info("\n验证集分类报告:\n" + classification_report(
            all_labels, all_preds, target_names=list(EMOTION_MAPPING.values()), digits=4
        ))
        
        val_loss = total_loss/total
        val_acc = correct/total*100
        logger.info(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
        return val_loss, val_acc

    def train(self, model_save_path):
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        logger.info(f"开始训练模型，保存路径: {model_save_path}")
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "best_acc": self.best_acc,
                    "hyperparams": self.hyperparams,
                    "class_weights": self.class_weights.cpu().numpy()
                }, model_save_path)
                logger.info(f"保存最佳模型（准确率{val_acc:.2f}%）到 {model_save_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"早停触发！最佳验证准确率: {self.best_acc:.2f}%")
                    break
        
        return self.best_acc

# ==================== 5. 预测器（修复遍历逻辑） ====================
class EnsemblePredictor:
    def __init__(self, model_paths: list):
        self.model_paths = model_paths
        self.models = []
        # 加载所有模型
        for idx, model_path in enumerate(model_paths):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model = EnhancedEmotionCNN(
                num_classes=6,
                dropout_rate=checkpoint["hyperparams"]["dropout_rate"]
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.models.append(model)
            logger.info(f"加载模型 {idx+1}/{len(model_paths)}: {model_path} | 最佳准确率: {checkpoint['best_acc']:.2f}%")
        # TTA变换
        self.tta_transforms = DataAugmentation.get_tta_transforms()

    def tta_predict(self, image):
        """单张图片TTA预测"""
        outputs = []
        with torch.no_grad():
            for transform in self.tta_transforms:
                # 确保图片是灰度图且变换正确
                img = transform(image).unsqueeze(0).to(device)
                for model in self.models:
                    out = model(img)
                    outputs.append(F.softmax(out, dim=1).cpu().numpy())
        # 多模型+TTA 均值
        outputs = np.array(outputs).mean(axis=0)
        return np.argmax(outputs)

    def predict_batch(self, test_dir: str, output_csv: str = "./results/test_predictions_ensemble.csv"):
        os.makedirs('./results', exist_ok=True)
        # 加载测试集（预过滤无效图片）
        test_dataset = EmotionGrayDataset(test_dir, is_train=False)
        if len(test_dataset) == 0:
            logger.error("测试集无有效图片！")
            return
        
        results = []
        # 修复遍历逻辑：用range遍历索引
        pbar = tqdm(range(len(test_dataset)), desc="多模型+TTA预测",colour='yellow')
        for idx in pbar:
            # 手动加载图片（避免transform干扰）
            img_path = test_dataset.image_paths[idx]
            img_name = test_dataset.__getitem__(idx)[2]  # 带后缀的文件名
            # 重新打开图片（确保TTA处理的是原始图片）
            with Image.open(img_path) as img:
                img = img.convert('RGB').convert('L')  # 强制灰度
                img = img.copy()
            
            # TTA+多模型预测
            pred = self.tta_predict(img)
            results.append({"ID": img_name, "Emotion": pred})
        
        # 保存CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        logger.info(f"预测CSV已保存: {output_csv} | 共{len(df)}条记录")
        return df

# ==================== 6. 主流程 ====================
def main():
    args = parse_args()
    config = TrainConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir
    )
    os.makedirs(config.model_save_dir, exist_ok=True)
    
    # 第一步：超参数调优
    logger.info("="*50 + "\n开始超参数调优")
    study = optuna.create_study(direction="maximize", study_name="fer_hyperopt")
    def objective(trial):
        hyperparams = suggest_hyperparams(trial)
        set_seed(42)
        try:
            data_manager = DataLoaderManager(
                config.train_dir,
                batch_size=hyperparams["batch_size"],
                val_split=config.val_split,
                dropout_rate=hyperparams["dropout_rate"]
            )
            model = create_model(num_classes=6, dropout_rate=hyperparams["dropout_rate"])
            trainer = Trainer(
                model, data_manager.train_loader, data_manager.val_loader,
                config, data_manager.class_weights, hyperparams
            )
            best_acc = trainer.train(f"./models/trial_{trial.number}_temp.pth")
            return best_acc
        except Exception as e:
            logger.error(f"超参数调优失败: {e}")
            return 0.0
    
    # 运行超参数搜索（5次试验）
    study.optimize(objective, n_trials=5, timeout=3600)
    best_trial = study.best_trial
    logger.info(f"最佳超参数: {best_trial.params} | 最佳验证准确率: {best_trial.value:.2f}%")
    
    # 第二步：训练多模型（不同种子）
    logger.info("="*50 + "\n开始训练多模型")
    model_paths = []
    for idx, seed in enumerate(config.seed_list):
        set_seed(seed)
        model_save_path = f"{config.model_save_dir}/emotion_model_{seed}.pth"
        # 加载数据
        data_manager = DataLoaderManager(
            config.train_dir,
            batch_size=best_trial.params.get("batch_size", 64),
            val_split=config.val_split,
            dropout_rate=best_trial.params.get("dropout_rate", 0.3)
        )
        # 创建模型
        model = create_model(
            num_classes=6,
            dropout_rate=best_trial.params.get("dropout_rate", 0.3)
        )
        # 训练
        trainer = Trainer(
            model, data_manager.train_loader, data_manager.val_loader,
            config, data_manager.class_weights, best_trial.params
        )
        trainer.train(model_save_path)
        model_paths.append(model_save_path)
    
    # 第三步：多模型+TTA融合预测
    logger.info("="*50 + "\n开始多模型+TTA融合预测")
    if len(model_paths) > 0:
        predictor = EnsemblePredictor(model_paths)
        predictor.predict_batch(config.test_dir)
    else:
        logger.error("无训练好的模型，跳过预测")
    
    logger.info("\n全部流程完成！结果文件在 ./results 目录下")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("程序被手动中断")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
