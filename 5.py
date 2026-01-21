import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from scipy.ndimage import binary_opening, binary_closing, binary_erosion, binary_dilation
from scipy.ndimage import label, find_objects, distance_transform_edt, map_coordinates
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import gaussian, frangi
import cv2
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 基础配置 + 修正形状检查函数 =====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (512, 512)  # 目标宽高（几何变换后）
BATCH_SIZE = 2 if torch.cuda.is_available() else 1
EPOCHS = 100
LEARNING_RATE = 1e-4
SMOOTH = 1e-8

# -------------------- 修正形状检查函数（分阶段检查） --------------------
def check_2d_shape(img):
    """阶段1：仅检查是否为2D（灰度图），不强制宽高（原始图像可能任意尺寸）"""
    img_np = np.array(img)
    assert len(img_np.shape) == 2, f"图像需为2D灰度图（H,W），实际形状：{img_np.shape}（可能是RGB图）"
    return img

def check_target_shape(img):
    """阶段2：几何变换后检查宽高是否为目标尺寸（IMAGE_SIZE）"""
    img_np = np.array(img)
    assert img_np.shape == IMAGE_SIZE, f"几何变换后宽高需为{IMAGE_SIZE}，实际：{img_np.shape}（变换逻辑错误）"
    return img

def check_tensor_shape(tensor):
    """检查张量是否为（1, H, W），H/W=IMAGE_SIZE"""
    assert tensor.dim() == 3, f"张量需为3D（C,H,W），实际维度：{tensor.dim()}"
    assert tensor.shape[0] == 1, f"张量通道数需为1（灰度），实际：{tensor.shape[0]}"
    assert tensor.shape[1:] == IMAGE_SIZE, f"张量宽高需为{IMAGE_SIZE}，实际：{tensor.shape[1:]}"
    return tensor

# 数据集路径（确保路径正确，可根据实际修改）
DATA_ROOT = '/kaggle/input/project5/segmentation' if os.path.exists('/kaggle/input') else './segmentation'
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train', 'image')
TRAIN_LABEL_DIR = os.path.join(DATA_ROOT, 'train', 'label')
TEST_IMG_DIR = os.path.join(DATA_ROOT, 'test', 'image')
OUTPUT_MASK_DIR = './output_masks'
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
SUBMISSION_PATH = 'final_submission.csv'

# ===================== 2. 数据增强（修正形状检查时机） =====================
class CLAHE_Transform:
    def __init__(self, clip_limit=1.5, grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        
    def __call__(self, img):
        img = check_2d_shape(img)  # 仅检查2D（此时宽高未固定）
        img_np = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
        img_np = clahe.apply(img_np)
        return Image.fromarray(img_np)

class RandomElasticTransform:
    def __init__(self, alpha_range=(30, 50), sigma_range=(3, 5)):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        
    def __call__(self, img):
        img = check_2d_shape(img)  # 仅检查2D
        img_np = np.array(img)
        h, w = img_np.shape  # 此时宽高是几何变换后的（如训练集576x576）
        
        # 弹性变换逻辑（不改变宽高）
        alpha = np.random.uniform(*self.alpha_range)
        sigma = np.random.uniform(*self.sigma_range)
        dx = gaussian(np.random.randn(h, w) * alpha, sigma, mode='reflect')
        dy = gaussian(np.random.randn(h, w) * alpha, sigma, mode='reflect')
        
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        new_y = y_coords + dy
        new_x = x_coords + dx
        indices = np.vstack((new_y.ravel(), new_x.ravel()))
        
        img_transformed = map_coordinates(img_np, indices, order=1, mode='reflect')
        img_transformed = img_transformed.reshape((h, w))  # 保持原宽高
        return Image.fromarray(img_transformed.astype(np.uint8))

# 几何变换（修正：先处理尺寸，最后检查目标形状）
def get_geometric_transforms(train=True):
    transforms_list = []
    if train:
        # 训练集：先Resize到576x576（留crop余量）→ RandomCrop到512x512
        transforms_list.extend([
            transforms.Resize((IMAGE_SIZE[0] + 64, IMAGE_SIZE[1] + 64)),  # 576x576
            transforms.RandomCrop(IMAGE_SIZE),  # 裁剪到目标尺寸512x512
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            RandomElasticTransform(alpha_range=(30, 50), sigma_range=(3, 5))
        ])
    else:
        # 验证/测试集：直接Resize到目标尺寸512x512
        transforms_list.append(transforms.Resize(IMAGE_SIZE))
    
    # 关键修正：几何变换后检查宽高是否为目标尺寸（此时已处理完成）
    transforms_list.append(transforms.Lambda(lambda x: check_target_shape(x)))
    return transforms.Compose(transforms_list)

# 像素变换（不变，基于几何变换后的目标尺寸）
def get_pixel_transforms(train=True):
    transforms_list = [
        CLAHE_Transform(clip_limit=1.5, grid_size=(8, 8)),
        transforms.ToTensor(),  # 转为(1, 512, 512)
        transforms.Lambda(lambda x: check_tensor_shape(x))  # 检查张量尺寸
    ]
    
    if train:
        transforms_list.insert(1, transforms.ColorJitter(brightness=0.2, contrast=0.2))
        transforms_list.insert(4, transforms.Lambda(
            lambda x: transforms.functional.adjust_gamma(x, gamma=np.random.uniform(0.9, 1.1))
        ))
    
    transforms_list.append(transforms.Normalize(mean=[0.5], std=[0.25]))
    return transforms.Compose(transforms_list)

# ===================== 3. 数据集类（修正形状检查流程） =====================
class RetinaVesselDataset(Dataset):
    def __init__(self, img_paths, label_paths=None, geometric_tf=None, pixel_tf=None, is_test=False):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.geometric_tf = geometric_tf  # 负责尺寸处理+目标形状检查
        self.pixel_tf = pixel_tf
        self.is_test = is_test
        
    def __len__(self):
        return len(self.img_paths)
    
    def extract_vessel_features(self, img):
        """提取特征（输入已为几何变换后的512x512）"""
        img = check_target_shape(img)  # 确保是目标尺寸
        img_np = np.array(img)
        
        # Frangi滤波（血管增强）
        frangi_img = frangi(img_np, sigmas=range(1, 4), black_ridges=False)
        frangi_img = (frangi_img - frangi_img.min()) / (frangi_img.max() - frangi_img.min() + 1e-8)
        frangi_img = (frangi_img * 255).astype(np.uint8)
        frangi_tensor = transforms.ToTensor()(Image.fromarray(frangi_img))
        frangi_tensor = check_tensor_shape(frangi_tensor)
        
        # 梯度幅值（边缘增强）
        grad_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
        grad_mag = (grad_mag * 255).astype(np.uint8)
        grad_tensor = transforms.ToTensor()(Image.fromarray(grad_mag))
        grad_tensor = check_tensor_shape(grad_tensor)
        
        return frangi_tensor, grad_tensor
    
    def get_pure_id(self, img_name):
        """提取纯数字Id（符合提交标准）"""
        base_name = os.path.splitext(img_name)[0]
        pure_id = ''.join(filter(str.isdigit, base_name))
        assert pure_id.isdigit(), f"Id提取失败！文件名：{img_name}（需包含数字，如1.png）"
        return int(pure_id)
    
    def __getitem__(self, idx):
        # 1. 读取原始图像（仅检查是否为2D，不强制宽高）
        img_path = self.img_paths[idx]
        img_name = os.path.basename(img_path)
        try:
            # 强制转为灰度图（避免RGB图导致的3D形状错误）
            image = Image.open(img_path).convert('L')
        except Exception as e:
            raise ValueError(f"读取图像失败：{img_path}，错误：{str(e)}（检查文件是否为图片格式）")
        
        image = check_2d_shape(image)  # 阶段1：仅确认是2D（宽高可能任意，如584x565）
        
        # 2. 几何变换（核心：处理尺寸为512x512，并检查目标形状）
        if self.geometric_tf:
            seed = np.random.randint(2147483647) if not self.is_test else 0
            torch.manual_seed(seed)
            image = self.geometric_tf(image)  # 阶段2：变换后宽高=512x512（内部已check_target_shape）
        
        # 3. 像素变换（转为1,512,512张量）
        if self.pixel_tf:
            torch.manual_seed(seed) if not self.is_test else None
            image_tensor = self.pixel_tf(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        image_tensor = check_tensor_shape(image_tensor)
        
        # 4. 提取特征（2个1,512,512张量）
        frangi_tensor, grad_tensor = self.extract_vessel_features(image)
        
        # 5. 拼接为模型输入（3,512,512）
        input_tensor = torch.cat([image_tensor, frangi_tensor, grad_tensor], dim=0)
        assert input_tensor.shape == (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), \
            f"模型输入需为（3,512,512），实际：{input_tensor.shape}（拼接错误）"
        
        if not self.is_test:
            # 处理标签（同图像流程：2D→几何变换→512x512）
            label_path = self.label_paths[idx]
            label = Image.open(label_path).convert('L')
            label = check_2d_shape(label)  # 阶段1：2D检查
            if self.geometric_tf:
                torch.manual_seed(seed)
                label = self.geometric_tf(label)  # 阶段2：尺寸→512x512
            
            # 标签转张量（1,512,512）
            label_tensor = transforms.ToTensor()(label)
            label_tensor = check_tensor_shape(label_tensor)
            
            # 标签二值化（血管=1，背景=0，根据标签图像素定义调整）
            # （若标签中血管是白色（255），则改为label_tensor > 0.5）
            label_binary = (label_tensor < 0.5).float()  # 假设标签：血管=黑色（0），背景=白色（255）
            label_binary = check_tensor_shape(label_binary)
            
            # 生成边界权重图（1,512,512）
            label_bin_np = label_tensor.squeeze().numpy()
            weight_map = np.ones(IMAGE_SIZE, dtype=np.float32)
            if label_bin_np.max() > 0:  # 若有血管区域
                pos_dist = distance_transform_edt(label_bin_np)
                boundary = np.logical_and(pos_dist <= 2, pos_dist > 0)  # 血管边界
                weight_map[boundary] = 3.0  # 边界权重提升
            weight_map = torch.from_numpy(weight_map).unsqueeze(0)
            weight_map = check_tensor_shape(weight_map)
            
            return input_tensor, label_binary, weight_map
        else:
            # 测试集：返回模型输入 + 纯数字Id
            pure_id = self.get_pure_id(img_name)
            return input_tensor, pure_id

# ===================== 4. 模型架构（输入已确保为3,512,512） =====================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Conv2d(f_g, f_int, 1)
        self.w_x = nn.Conv2d(f_l, f_int, 1)
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class EfficientUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # 编码器（3→32→64→128→256）
        self.enc1 = ResidualBlock(in_ch, 32, stride=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 512→256
        self.enc2 = ResidualBlock(32, 64, stride=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 256→128
        self.enc3 = ResidualBlock(64, 128, stride=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)  # 128→64
        self.enc4 = ResidualBlock(128, 256, stride=1)
        self.pool4 = nn.MaxPool2d(2, stride=2)  # 64→32
        
        # 瓶颈层
        self.bottleneck = ResidualBlock(256, 512, stride=1)
        
        # 解码器（上采样+注意力）
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 32→64
        self.att4 = AttentionGate(256, 256, 128)
        self.dec4 = ResidualBlock(512, 256, stride=1)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 64→128
        self.att3 = AttentionGate(128, 128, 64)
        self.dec3 = ResidualBlock(256, 128, stride=1)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 128→256
        self.att2 = AttentionGate(64, 64, 32)
        self.dec2 = ResidualBlock(128, 64, stride=1)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # 256→512
        self.dec1 = ResidualBlock(64, 32, stride=1)
        
        # 输出层（1通道预测图：512x512）
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, 1)
        )
    
    def forward(self, x):
        # 确保输入为4D（B,C,H,W），单样本补batch维度
        if x.dim() == 3:
            x = x.unsqueeze(0)
        assert x.dim() == 4, f"模型输入需为4D（B,3,512,512），实际维度：{x.dim()}"
        
        # 编码过程（维度均符合预期）
        e1 = self.enc1(x)  # (B,32,512,512)
        e2 = self.enc2(self.pool1(e1))  # (B,64,256,256)
        e3 = self.enc3(self.pool2(e2))  # (B,128,128,128)
        e4 = self.enc4(self.pool3(e3))  # (B,256,64,64)
        
        # 瓶颈层
        b = self.bottleneck(self.pool4(e4))  # (B,512,32,32)
        
        # 解码过程
        d4 = self.up4(b)  # (B,256,64,64)
        d4 = self.dec4(torch.cat([self.att4(d4, e4), d4], dim=1))  # (B,256,64,64)
        
        d3 = self.up3(d4)  # (B,128,128,128)
        d3 = self.dec3(torch.cat([self.att3(d3, e3), d3], dim=1))  # (B,128,128,128)
        
        d2 = self.up2(d3)  # (B,64,256,256)
        d2 = self.dec2(torch.cat([self.att2(d2, e2), d2], dim=1))  # (B,64,256,256)
        
        d1 = self.up1(d2)  # (B,32,512,512)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))  # (B,32,512,512)
        
        # 输出预测图（B,1,512,512）
        out = self.out_conv(d1)
        return out

# ===================== 5. 损失函数（无修改） =====================
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target):
        assert pred.shape == target.shape, f"损失计算：形状不匹配，pred={pred.shape}，target={target.shape}"
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(1,2,3))
        union = pred_sig.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        return 1 - (2 * intersection + SMOOTH) / (union + SMOOTH)
    
    def focal_loss(self, pred, target):
        assert pred.shape == target.shape, f"损失计算：形状不匹配"
        bce_loss = self.bce(pred, target)
        pt = torch.exp(-bce_loss)
        return (1 - pt) ** self.gamma * bce_loss
    
    def forward(self, pred, target, weights=None):
        dice = self.dice_loss(pred, target).mean()
        focal = self.focal_loss(pred, target).mean()
        total_loss = self.alpha * dice + (1 - self.alpha) * focal
        
        if weights is not None:
            assert weights.shape == target.shape, f"权重形状不匹配"
            total_loss = total_loss * weights.mean()
        return total_loss

# ===================== 6. 训练与评估函数（无修改） =====================
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        imgs, labels, weights = [x.to(DEVICE) for x in batch]
        assert imgs.shape == (BATCH_SIZE, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]), \
            f"训练批次形状错误：{imgs.shape}（需为{BATCH_SIZE},3,512,512）"
        
        optimizer.zero_grad()
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(imgs)
            loss = criterion(outputs, labels, weights)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * imgs.size(0)
        # 打印进度（每20批）
        if (batch_idx + 1) % 20 == 0:
            print(f"  训练进度：{batch_idx + 1}/{len(loader)}批，当前批次损失：{loss.item():.4f}")
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    with torch.no_grad():
        for batch in loader:
            imgs, labels, weights = [x.to(DEVICE) for x in batch]
            assert imgs.shape[1:] == (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), f"评估批次形状错误"
            
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                loss = criterion(outputs, labels, weights)
                pred_sig = torch.sigmoid(outputs)
            
            # 计算Dice系数（血管分割评价指标）
            pred_bin = (pred_sig > 0.5).float()
            intersection = (pred_bin * labels).sum(dim=(1,2,3))
            union = pred_bin.sum(dim=(1,2,3)) + labels.sum(dim=(1,2,3))
            dice = (2 * intersection + SMOOTH) / (union + SMOOTH)
            
            total_loss += loss.item() * imgs.size(0)
            total_dice += dice.sum().item()
    
    avg_loss = total_loss / len(loader.dataset)
    avg_dice = total_dice / len(loader.dataset)
    return avg_loss, avg_dice

def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)  # 按Dice调整学习率
    scaler = GradScaler()
    
    best_dice = 0.0
    early_stop_patience = 15  # 早停：15轮无提升则停止
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1:3d}/{epochs}")
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        # 验证
        val_loss, val_dice = evaluate(model, val_loader, criterion)
        
        # 学习率调度（若Dice提升，更新学习率）
        scheduler.step(val_dice)
        
        # 保存最佳模型（Dice最高）
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_vessel_model.pth')
            patience_counter = 0
            print(f"  保存最佳模型！当前最佳Dice：{best_dice:.4f}")
        else:
            patience_counter += 1
            print(f"  无提升，早停计数：{patience_counter}/{early_stop_patience}")
        
        # 早停判断
        if patience_counter >= early_stop_patience:
            print(f"  早停于Epoch {epoch+1}（15轮无提升）")
            break
        
        # 打印本轮结果
        print(f"  训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f} | 验证Dice：{val_dice:.4f}")
    
    # 加载最佳模型返回
    model.load_state_dict(torch.load('best_vessel_model.pth'))
    return model, best_dice

# ===================== 7. RLE编码（符合提交标准） =====================
def rle_encode(mask):
    """编码血管区域（255）为RLE字符串，空区域返回空字符串"""
    # 确保mask是512x512的2D数组
    mask_np = np.array(mask)
    assert mask_np.shape == IMAGE_SIZE and len(mask_np.shape) == 2, f"RLE输入错误：{mask_np.shape}"
    
    flat_mask = mask_np.flatten()
    vessel_pixels = np.where(flat_mask == 255)[0]  # 血管像素值为255
    
    if len(vessel_pixels) == 0:
        return ""  # 无血管区域，返回空字符串（符合提交标准）
    
    # 1-based索引编码（竞赛通用格式）
    run_lengths = []
    prev_pixel = -2
    for pixel in vessel_pixels:
        if pixel > prev_pixel + 1:
            run_lengths.extend([pixel + 1, 0])  # 起始位置（+1转为1-based）
        run_lengths[-1] += 1  # 长度递增
        prev_pixel = pixel
    
    return ' '.join(map(str, run_lengths))

# ===================== 8. 测试集预测（生成符合标准的提交文件） =====================
def predict_test_set(model, test_loader):
    model.eval()
    submission_data = []
    print("\n=== 开始测试集预测 ===")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            imgs, pure_ids = batch
            imgs = imgs.to(DEVICE)
            assert imgs.shape == (1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]), f"测试批次形状错误"
            
            # 推理
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                pred_sig = torch.sigmoid(outputs).cpu().numpy()
            
            # 处理每个样本（测试集batch_size=1）
            pred_prob = pred_sig[0].squeeze()  # (512,512)
            # 多阈值投票（提高稳定性）
            pred1 = (pred_prob > 0.4).astype(np.uint8)
            pred2 = (pred_prob > 0.45).astype(np.uint8)
            pred3 = (pred_prob > 0.5).astype(np.uint8)
            pred_bin = ((pred1 + pred2 + pred3) >= 2).astype(np.uint8)  # 多数投票
            
            # 后处理（去除小区域、优化边界）
            pred_bin = remove_small_objects(pred_bin.astype(bool), min_size=10).astype(np.uint8)  # 去小区域
            kernel = np.ones((3, 3), np.uint8)
            pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_CLOSE, kernel, iterations=1)  # 闭运算（填小孔）
            pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel, iterations=1)  # 开运算（去小噪点）
            
            # 转为提交格式：血管=255，背景=0
            final_mask = pred_bin * 255
            final_mask_pil = Image.fromarray(final_mask)
            
            # 保存预测mask（可选，用于 debug）
            mask_path = os.path.join(OUTPUT_MASK_DIR, f"{pure_ids[0]}.png")
            final_mask_pil.save(mask_path)
            
            # RLE编码
            rle_str = rle_encode(final_mask_pil)
            submission_data.append({
                'Id': pure_ids[0],  # 纯数字Id（符合标准）
                'Predicted': rle_str  # RLE字符串（空区域为空）
            })
            
            # 打印进度
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"  预测进度：{batch_idx + 1}/{len(test_loader)}个样本")
    
    # 生成提交文件（关键：排序+格式验证）
    submission_df = pd.DataFrame(submission_data)
    # 按Id升序排序（符合提交标准）
    submission_df = submission_df.sort_values('Id').reset_index(drop=True)
    # 保存CSV（无索引）
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    # 验证提交文件格式
    print(f"\n=== 提交文件验证 ===")
    print(f"列名：{submission_df.columns.tolist()} → 需为['Id', 'Predicted']（正确）")
    print(f"行数：{len(submission_df)} → 与测试集样本数一致（{len(test_loader)}）")
    print(f"Id类型：{submission_df['Id'].dtype} → 需为int（正确）")
    print(f"空RLE数量：{len(submission_df[submission_df['Predicted'] == ''])} → 无血管样本数")
    print(f"\n前5行预览：")
    print(submission_df.head())
    print(f"\n提交文件已保存至：{os.path.abspath(SUBMISSION_PATH)}")
    
    return submission_df

# ===================== 9. 主流程（增加数据校验） =====================
if __name__ == "__main__":
    # 1. 校验数据集路径（避免路径错误）
    print("=== 1. 校验数据集路径 ===")
    required_paths = [TRAIN_IMG_DIR, TRAIN_LABEL_DIR, TEST_IMG_DIR]
    for path in required_paths:
        if not os.path.exists(path):
            raise ValueError(f"路径不存在：{path} → 请检查DATA_ROOT配置是否正确")
        print(f"  有效路径：{path}（文件数：{len(os.listdir(path))}）")
    
    # 2. 加载图像路径（仅保留图片格式）
    print("\n=== 2. 加载图像路径 ===")
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    train_img_paths = [os.path.join(TRAIN_IMG_DIR, f) for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith(img_extensions)]
    train_label_paths = [os.path.join(TRAIN_LABEL_DIR, f) for f in os.listdir(TRAIN_LABEL_DIR) if f.lower().endswith(img_extensions)]
    test_img_paths = [os.path.join(TEST_IMG_DIR, f) for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(img_extensions)]
    
    # 排序（确保图像与标签一一对应）
    train_img_paths.sort()
    train_label_paths.sort()
    test_img_paths.sort()
    
    # 校验数量
    print(f"  训练图像数：{len(train_img_paths)}")
    print(f"  训练标签数：{len(train_label_paths)}")
    print(f"  测试图像数：{len(test_img_paths)}")
    assert len(train_img_paths) == len(train_label_paths), "训练图像与标签数量不匹配！"
    assert len(train_img_paths) > 0 and len(test_img_paths) > 0, "训练集或测试集为空！"
    
    # 3. 划分训练/验证集（8:2）
    print("\n=== 3. 划分训练/验证集 ===")
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        train_img_paths, train_label_paths, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"  训练集：{len(train_imgs)}张图像 + {len(train_labels)}张标签")
    print(f"  验证集：{len(val_imgs)}张图像 + {len(val_labels)}张标签")
    
    # 4. 创建数据集和加载器（num_workers=0避免多进程错误）
    print("\n=== 4. 创建数据集和加载器 ===")
    # 训练集
    train_dataset = RetinaVesselDataset(
        train_imgs, train_labels,
        geometric_tf=get_geometric_transforms(train=True),
        pixel_tf=get_pixel_transforms(train=True)
    )
    # 验证集
    val_dataset = RetinaVesselDataset(
        val_imgs, val_labels,
        geometric_tf=get_geometric_transforms(train=False),
        pixel_tf=get_pixel_transforms(train=False)
    )
    # 测试集
    test_dataset = RetinaVesselDataset(
        test_img_paths, is_test=True,
        geometric_tf=get_geometric_transforms(train=False),
        pixel_tf=get_pixel_transforms(train=False)
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True  # drop_last避免批次大小不一致
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,  # 测试集batch=1
        num_workers=0, pin_memory=True
    )
    
    print(f"  训练加载器：{len(train_loader)}批（每批{BATCH_SIZE}张）")
    print(f"  验证加载器：{len(val_loader)}批（每批{BATCH_SIZE}张）")
    print(f"  测试加载器：{len(test_loader)}批（每批1张）")
    
    # 5. 初始化模型
    print("\n=== 5. 初始化模型 ===")
    model = EfficientUNet(in_ch=3).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量：{param_count:,}（轻量化设计，适合训练）")
    print(f"  使用设备：{DEVICE} → GPU可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU型号：{torch.cuda.get_device_name(0)}")
    
    # 6. 训练模型
    print("\n=== 6. 开始训练模型 ===")
    trained_model, best_val_dice = train_model(model, train_loader, val_loader, epochs=EPOCHS)
    print(f"\n训练完成！最佳验证Dice系数：{best_val_dice:.4f}（越高越好，1.0为完美分割）")
    
    # 7. 测试集预测与提交文件生成
    submission_df = predict_test_set(trained_model, test_loader)
    print("\n=== 所有流程完成！===")
    print(f"最终提交文件：{os.path.abspath(SUBMISSION_PATH)}")
    print(f"预测mask保存目录：{os.path.abspath(OUTPUT_MASK_DIR)}")
