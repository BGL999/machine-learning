"""
植物分类深度学习系统 - 特征学习方法
Colab专用版本（无命令行参数）
CSV输出格式：ID, Category（保持原始顺序）
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# ==============================
# 全局配置 - 已适配Kaggle路径
# ==============================
class Config:
    """配置类，存储所有超参数和路径"""
    # Kaggle数据路径（已设置为您提供的路径）
    TRAIN_DIR = '/kaggle/input/poject1/dataset-for-task1/dataset-for-task1/train'
    TEST_DIR = '/kaggle/input/project2-test'
    
    # 输出文件路径（保存到Kaggle工作目录）
    OUTPUT_CSV = '/kaggle/working/predictions.csv'
    MODEL_SAVE_PATH = '/kaggle/working/best_plant_model.pth'
    TRAINING_HISTORY_PATH = '/kaggle/working/training_history.png'
    CONFUSION_MATRIX_PATH = '/kaggle/working/confusion_matrix.png'
    VISUALIZATION_PATH = '/kaggle/working/prediction_visualization.png'
    
    # 训练参数
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001
    PATIENCE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 图像参数
    IMG_SIZE = 224
    NUM_WORKERS = 2
    
    # 随机种子
    SEED = 42

# 设置随机种子
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ==============================
# 数据集类
# ==============================
class PlantTrainDataset(Dataset):
    """训练数据集类"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 获取所有类别
        self.categories = sorted([d for d in os.listdir(root_dir) 
                                 if os.path.isdir(os.path.join(root_dir, d))])
        self.num_classes = len(self.categories)
        
        # 创建类别映射
        self.class_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_class = {idx: cat for idx, cat in enumerate(self.categories)}
        
        # 收集图像和标签
        self.image_paths = []
        self.labels = []
        
        for class_name in self.categories:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
        
        print(f"训练集总样本数: {len(self.image_paths)}")
        print(f"类别数量: {self.num_classes}")
        
        # 统计类别分布
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """打印类别分布"""
        class_counts = {cat: 0 for cat in self.categories}
        for label in self.labels:
            class_counts[self.idx_to_class[label]] += 1
        
        print("各类别样本数量:")
        for cat, count in class_counts.items():
            print(f"  {cat}: {count}张")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_mapping(self):
        """获取类别映射"""
        return self.idx_to_class


class PlantTestDataset(Dataset):
    """测试数据集类"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 收集测试图像 - 保持原始顺序
        self.image_paths = []
        self.image_names = []
        
        # 使用sorted确保顺序一致性，但保持按文件名顺序
        for img_name in sorted(os.listdir(root_dir)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root_dir, img_name)
                self.image_paths.append(img_path)
                self.image_names.append(img_name)
        
        print(f"测试集图片数量: {len(self.image_paths)}")
        
        # 显示前5个文件名，确认顺序
        print(f"前5个测试图片: {self.image_names[:5]}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name

# ==============================
# 数据转换
# ==============================
def get_train_transforms():
    """获取训练数据转换"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                              saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    """获取验证/测试数据转换"""
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

# ==============================
# 深度学习模型
# ==============================
class PlantClassificationModel(nn.Module):
    """植物分类深度学习模型"""
    def __init__(self, num_classes, use_pretrained=True):
        super(PlantClassificationModel, self).__init__()
        
        # 使用预训练的ResNet50
        self.backbone = models.resnet50(pretrained=use_pretrained)
        
        # 冻结部分层进行特征学习
        self._freeze_layers()
        
        # 替换分类头
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def _freeze_layers(self):
        """冻结网络层进行特征学习"""
        # 冻结所有层
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 解冻最后两层进行微调
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)

# ==============================
# 模型训练模块
# ==============================
class ModelTrainer:
    """模型训练器"""
    def __init__(self, train_dir, val_split=0.2):
        self.train_dir = train_dir
        self.val_split = val_split
        self.device = Config.DEVICE
        
        # 数据转换
        self.train_transform = get_train_transforms()
        self.val_transform = get_val_transforms()
        
        # 数据集
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # 模型
        self.model = None
        self.idx_to_class = None
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        
    def prepare_data(self):
        """准备数据"""
        print("准备训练数据...")
        
        # 检查训练目录是否存在
        if not os.path.exists(self.train_dir):
            print(f"错误：训练目录不存在: {self.train_dir}")
            print("请检查路径是否正确！")
            return None, None
        
        # 创建完整数据集
        self.full_dataset = PlantTrainDataset(self.train_dir, self.train_transform)
        self.idx_to_class = self.full_dataset.get_class_mapping()
        
        # 划分训练集和验证集
        train_size = int((1 - self.val_split) * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset, [train_size, val_size]
        )
        
        # 验证集使用不同的转换
        self.val_dataset.dataset.transform = self.val_transform
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=min(Config.NUM_WORKERS, os.cpu_count()//2),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=min(Config.NUM_WORKERS, os.cpu_count()//2),
            pin_memory=True
        )
        
        print(f"训练集样本数: {len(self.train_dataset)}")
        print(f"验证集样本数: {len(self.val_dataset)}")
        
        return self.train_loader, self.val_loader
    
    def build_model(self):
        """构建模型"""
        print("构建深度学习模型...")
        
        num_classes = self.full_dataset.num_classes
        self.model = PlantClassificationModel(num_classes, use_pretrained=True)
        self.model = self.model.to(self.device)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() 
                              if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
        return self.model
    
    def train(self, num_epochs=None, learning_rate=None):
        """训练模型"""
        if num_epochs is None:
            num_epochs = Config.NUM_EPOCHS
        if learning_rate is None:
            learning_rate = Config.LEARNING_RATE
        
        print(f"\n开始训练模型...")
        print(f"训练周期: {num_epochs}")
        print(f"学习率: {learning_rate}")
        print(f"设备: {self.device}")
        print("=" * 60)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # 训练循环
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [训练]')
            for images, labels in train_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * train_correct / train_total
                })
            
            avg_train_loss = train_loss / len(self.train_loader)
            train_acc = 100. * train_correct / train_total
            
            # 验证阶段
            val_loss, val_acc = self._validate()
            
            # 更新学习率
            scheduler.step()
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_mapping': self.idx_to_class,
                    'num_classes': self.full_dataset.num_classes
                }, Config.MODEL_SAVE_PATH)
                
                print(f"✓ 保存最佳模型，验证准确率: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # 记录训练过程
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 打印epoch结果
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"  最佳验证准确率: {self.best_val_acc:.2f}%")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 60)
            
            # 早停检查
            if patience_counter >= Config.PATIENCE:
                print(f"\n早停触发！连续 {Config.PATIENCE} 个epoch验证准确率未提升")
                break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"\n训练完成！最佳验证准确率: {self.best_val_acc:.2f}%")
        
        # 可视化训练过程
        self._plot_training_history()
        
        # 评估模型
        self.evaluate()
        
        return self.model
    
    def _validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return avg_val_loss, val_acc
    
    def evaluate(self):
        """评估模型性能"""
        print("\n评估模型性能...")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="评估中"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        
        print("\n" + "=" * 50)
        print("模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print("=" * 50)
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        self._plot_confusion_matrix(cm, list(self.idx_to_class.values()))
        
        # 分类报告
        print("\n详细分类报告:")
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=list(self.idx_to_class.values()), 
            digits=4
        )
        print(report)
        
        return accuracy
    
    def _plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(self.train_losses, label='训练损失', linewidth=2)
        axes[0].plot(self.val_losses, label='验证损失', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('损失', fontsize=12)
        axes[0].set_title('训练和验证损失曲线', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(self.val_accuracies, label='验证准确率', 
                    color='orange', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('准确率 (%)', fontsize=12)
        axes[1].set_title('验证准确率曲线', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Config.TRAINING_HISTORY_PATH, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrix(self, cm, class_names):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title('混淆矩阵', fontsize=16)
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(Config.CONFUSION_MATRIX_PATH, dpi=300, bbox_inches='tight')
        plt.show()

# ==============================
# 模型预测模块 - 保持原始顺序
# ==============================
class ModelPredictor:
    """模型预测器"""
    def __init__(self, model_path=None):
        self.model_path = model_path or Config.MODEL_SAVE_PATH
        self.device = Config.DEVICE
        self.model = None
        self.idx_to_class = None
        self.num_classes = None
        
    def load_model(self):
        """加载训练好的模型"""
        print(f"加载模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 获取类别信息
        self.idx_to_class = checkpoint['class_mapping']
        self.num_classes = checkpoint['num_classes']
        
        # 创建模型
        self.model = PlantClassificationModel(self.num_classes, use_pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载成功！验证准确率: {checkpoint.get('val_acc', 'N/A')}%")
        print(f"类别数量: {self.num_classes}")
        
        return self.model
    
    def predict(self, test_dir, output_csv=None):
        """对测试集进行预测，保持原始顺序"""
        if output_csv is None:
            output_csv = Config.OUTPUT_CSV
        
        print(f"\n对测试集进行预测...")
        print(f"测试集目录: {test_dir}")
        print(f"输出文件: {output_csv}")
        print("=" * 60)
        
        # 检查测试目录是否存在
        if not os.path.exists(test_dir):
            print(f"错误：测试目录不存在: {test_dir}")
            print("请检查路径是否正确！")
            return None
        
        # 数据转换
        test_transform = get_val_transforms()
        
        # 创建测试数据集
        test_dataset = PlantTestDataset(test_dir, test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,  # 非常重要！不能打乱顺序
            num_workers=min(Config.NUM_WORKERS, os.cpu_count()//2)
        )
        
        # 进行预测
        all_filenames = []
        all_predictions = []
        all_confidences = []
        
        with torch.no_grad():
            for images, filenames in tqdm(test_loader, desc="预测中"):
                images = images.to(self.device)
                outputs = self.model(images)
                
                # 获取概率和预测
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, 1)
                
                all_filenames.extend(filenames)
                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # 创建结果DataFrame - 保持原始顺序
        results = []
        print(f"\n开始生成CSV文件，保持原始顺序...")
        
        for filename, pred_idx, confidence in zip(all_filenames, all_predictions, all_confidences):
            pred_class = self.idx_to_class[pred_idx]
            # 去掉文件扩展名，只保留ID
            file_id = os.path.splitext(filename)[0]
            results.append({
                'ID': file_id,           # 列名改为ID，值去掉扩展名
                'Category': pred_class   # 列名改为Category
            })
        
        results_df = pd.DataFrame(results)
        # 重要：不进行排序，保持原始顺序
        # results_df = results_df.sort_values('ID')  # 这行已删除
        
        # 保存到CSV
        results_df.to_csv(output_csv, index=False)
        
        print(f"\n预测完成！")
        print(f"共预测了 {len(results_df)} 张图片")
        print(f"结果已保存到: {output_csv}")
        print(f"输出顺序保持原始测试集顺序")
        
        # 显示统计信息和顺序验证
        self._print_statistics(results_df)
        
        return results_df
    
    def _print_statistics(self, results_df):
        """打印预测统计信息"""
        print("\n预测结果统计:")
        print("=" * 40)
        
        # 各类别预测数量
        category_counts = results_df['Category'].value_counts()
        print("\n各类别预测数量:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}张")
        
        # 显示前10个预测结果的顺序
        print(f"\n前10个预测结果（保持原始顺序）:")
        print(results_df.head(10).to_string(index=False))
        
        # 验证顺序是否与文件名顺序一致
        if len(results_df) > 0:
            print(f"\n顺序验证:")
            print(f"  第一个预测ID: {results_df.iloc[0]['ID']}")
            print(f"  最后一个预测ID: {results_df.iloc[-1]['ID']}")
    
    def visualize_predictions(self, test_dir, results_df, num_samples=12):
        """可视化部分预测结果"""
        # 检查测试目录是否存在
        if not os.path.exists(test_dir):
            print(f"错误：测试目录不存在: {test_dir}")
            print("无法可视化预测结果")
            return
        
        # 取前num_samples个样本（保持顺序）
        sample_df = results_df.head(min(num_samples, len(results_df)))
        
        # 创建子图
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(sample_df.iterrows()):
            if idx >= len(axes):
                break
            
            # 由于我们只有ID没有扩展名，需要找到对应的图片文件
            # 查找测试目录中以该ID开头的文件
            img_files = [f for f in os.listdir(test_dir) 
                        if f.startswith(row['ID']) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not img_files:
                print(f"未找到ID为 {row['ID']} 的图片")
                continue
            
            img_path = os.path.join(test_dir, img_files[0])
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                print(f"无法加载图片: {img_path}")
                continue
            
            axes[idx].imshow(img)
            
            # 设置标题
            axes[idx].set_title(
                f"ID: {row['ID']}\nCategory: {row['Category']}", 
                fontsize=10,
                fontweight='bold'
            )
            
            axes[idx].axis('off')
        
        # 隐藏多余的子图
        for idx in range(len(sample_df), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('植物分类预测结果可视化（保持原始顺序）', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Config.VISUALIZATION_PATH, dpi=300, bbox_inches='tight')
        plt.show()

# ==============================
# 主执行函数
# ==============================
def run_plant_classification():
    """
    植物分类主函数 - 专门为Colab/Jupyter Notebook设计
    自动执行完整流程：训练+预测
    CSV输出格式：ID, Category（保持原始顺序）
    """
    print("=" * 70)
    print("植物分类深度学习系统 - Colab专用版本")
    print("=" * 70)
    print(f"训练数据: {Config.TRAIN_DIR}")
    print(f"测试数据: {Config.TEST_DIR}")
    print(f"输出文件: {Config.OUTPUT_CSV}")
    print(f"设备: {Config.DEVICE}")
    print(f"训练周期: {Config.NUM_EPOCHS}")
    print(f"CSV格式: ID, Category（保持原始顺序）")
    print("=" * 70)
    
    # 检查路径是否存在
    if not os.path.exists(Config.TRAIN_DIR):
        print(f"❌ 错误：训练目录不存在: {Config.TRAIN_DIR}")
        print("请确保Kaggle数据集已正确挂载")
        return None
    
    if not os.path.exists(Config.TEST_DIR):
        print(f"❌ 错误：测试目录不存在: {Config.TEST_DIR}")
        print("请确保Kaggle数据集已正确挂载")
        return None
    
    results_df = None
    
    try:
        # ==============================
        # 第一步：训练模型
        # ==============================
        print("\n" + "=" * 60)
        print("第一步：训练深度学习模型")
        print("=" * 60)
        
        # 创建训练器
        trainer = ModelTrainer(Config.TRAIN_DIR)
        
        # 准备数据
        print("正在准备训练数据...")
        train_loader, val_loader = trainer.prepare_data()
        
        if train_loader is None:
            print("❌ 数据准备失败，无法继续训练！")
            return None
        
        # 构建模型
        print("正在构建深度学习模型...")
        trainer.build_model()
        
        # 训练模型
        print("开始训练模型（这可能需要一些时间）...")
        model = trainer.train(num_epochs=Config.NUM_EPOCHS, 
                             learning_rate=Config.LEARNING_RATE)
        
        print("✅ 模型训练完成！")
        
        # ==============================
        # 第二步：进行预测
        # ==============================
        print("\n" + "=" * 60)
        print("第二步：对测试集进行预测（保持原始顺序）")
        print("=" * 60)
        
        # 创建预测器
        predictor = ModelPredictor()
        
        # 加载刚刚训练好的模型
        print("正在加载训练好的模型...")
        predictor.load_model()
        
        # 进行预测
        print("正在对测试集进行预测...")
        results_df = predictor.predict(Config.TEST_DIR, Config.OUTPUT_CSV)
        
        if results_df is not None:
            print("✅ 预测完成！")
            
            # 显示CSV文件内容预览
            print("\nCSV文件内容预览（前10行）:")
            try:
                with open(Config.OUTPUT_CSV, 'r') as f:
                    lines = f.readlines()[:11]  # 读取前10行数据（包括表头）
                    for line in lines:
                        print(line.strip())
            except:
                print("无法读取CSV文件")
            
        else:
            print("❌ 预测失败！")
            
    except Exception as e:
        print(f"\n❌ 程序执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n" + "=" * 70)
    print("🎉 植物分类系统执行完成！")
    print("=" * 70)
    print(f"📄 预测结果已保存到: {Config.OUTPUT_CSV}")
    print(f"🤖 训练好的模型已保存到: {Config.MODEL_SAVE_PATH}")
    print(f"📊 训练过程可视化: {Config.TRAINING_HISTORY_PATH}")
    print(f"📈 混淆矩阵: {Config.CONFUSION_MATRIX_PATH}")
    print(f"🖼️ 预测结果可视化: {Config.VISUALIZATION_PATH}")
    print("=" * 70)
    
    return results_df

# ==============================
# 程序入口 - 直接运行
# ==============================
if __name__ == "__main__":
    # 在Colab中直接运行完整流程
    print("开始执行植物分类深度学习系统...")
    results = run_plant_classification()
    
    if results is not None:
        print("\n✅ 所有任务已完成！")
        print(f"\n最终CSV文件格式（保持原始顺序）:")
        print("| ID | Category |")
        print("|----|----------|")
        # 显示前5个预测结果
        for i in range(min(5, len(results))):
            row = results.iloc[i]
            print(f"| {row['ID']} | {row['Category']} |")
    else:
        print("\n❌ 程序执行失败，请检查错误信息。")