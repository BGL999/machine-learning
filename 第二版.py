import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import os
import logging
import warnings
import random
import numpy as np
from tqdm import tqdm
import cv2
warnings.filterwarnings("ignore")

# ---------------------- 1. 配置 ----------------------
CONFIG = {
    "TRAIN_IMG_DIR": "/kaggle/input/project4project4/detection/train",
    "TRAIN_CSV_PATH": "/kaggle/input/project4project4/detection/fovea_localization_train_GT.csv",
    "TEST_IMG_DIR": "/kaggle/input/project4project4/detection/test",
    "OUTPUT_DIR": "/kaggle/working",
    "BATCH_SIZE": 8,
    "EPOCHS": 30,
    "LEARNING_RATE": 0.001,
    "WEIGHT_DECAY": 1e-4,
    "VAL_RATIO": 0.15,
    "PATIENCE": 6,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BACKBONE": "resnet18",
    "IMAGE_SIZE": 512,
    "RESUME_TRAIN": False,
    "CHECKPOINT_PATH": "/kaggle/working/best_model.pth",
    "IMAGE_SUFFIXES": [".jpg", ".png", ".jpeg"],
    "UNCERTAINTY_WEIGHTING": True,
    "TTA_NUM": 3,
}

# 配置日志
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CONFIG["OUTPUT_DIR"], "train.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------- 2. 数据集类（修复版） ----------------------
class FoveaDataset(Dataset):
    def __init__(self, img_dir, csv_path, image_size=512, is_train=True):
        self.img_dir = img_dir
        self.image_size = image_size
        self.is_train = is_train
        
        # 加载图片映射
        self.img_paths = {}
        for f in os.listdir(img_dir):
            if any(f.endswith(ext) for ext in CONFIG["IMAGE_SUFFIXES"]):
                base_name = os.path.splitext(f)[0]
                self.img_paths[base_name] = os.path.join(img_dir, f)
        
        # 加载CSV标签
        df = pd.read_csv(csv_path)
        df["data"] = df["data"].astype(int).astype(str).str.zfill(4)
        
        # 过滤有效样本
        self.samples = []
        for _, row in df.iterrows():
            img_base = row["data"].strip()
            if img_base in self.img_paths:
                x, y = float(row["Fovea_X"]), float(row["Fovea_Y"])
                if x >= 0 and y >= 0:
                    self.samples.append({
                        "img_path": self.img_paths[img_base],
                        "coords": (x, y),
                        "visible": 1.0
                    })
        
        logger.info(f"数据集加载完成: {len(self.samples)}个样本 (训练模式: {is_train})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        coords = sample["coords"]
        visible = sample["visible"]
        
        # 读取图像
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        # 数据增强（仅在训练时）
        if self.is_train and random.random() < 0.5:
            # 随机水平翻转
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            coords = (orig_w - coords[0] - 1, coords[1])
        
        # 计算缩放比例
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        
        # 缩放图像
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # 缩放坐标
        scaled_x = coords[0] * scale_x
        scaled_y = coords[1] * scale_y
        
        # 归一化坐标到[0,1]
        norm_x = scaled_x / self.image_size
        norm_y = scaled_y / self.image_size
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image)
        
        return {
            "image": image_tensor,
            "coords": torch.tensor([norm_x, norm_y], dtype=torch.float32),
            "visible": torch.tensor([visible], dtype=torch.float32),
            "img_name": os.path.basename(img_path),
            "orig_size": torch.tensor([orig_w, orig_h], dtype=torch.float32)
        }

# ---------------------- 3. 模型（简化稳定版） ----------------------
class FoveaModel(nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super().__init__()
        
        # 加载预训练骨干网络
        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=True)
            num_features = backbone.fc.in_features
        else:
            raise ValueError(f"不支持的骨干网络: {backbone_name}")
        
        # 移除最后的全连接层
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        # 回归头（预测归一化坐标）
        self.reg_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Sigmoid()  # 输出在[0,1]范围内
        )
        
        # 分类头（预测是否可见）
        self.cls_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        reg_out = self.reg_head(features)
        cls_out = self.cls_head(features)
        return reg_out, cls_out

# ---------------------- 4. 训练函数（修复错误） ----------------------
def train_model():
    logger.info("开始训练...")
    
    # 创建数据集
    full_dataset = FoveaDataset(
        img_dir=CONFIG["TRAIN_IMG_DIR"],
        csv_path=CONFIG["TRAIN_CSV_PATH"],
        image_size=CONFIG["IMAGE_SIZE"],
        is_train=True
    )
    
    # 划分训练集和验证集
    val_size = max(1, int(CONFIG["VAL_RATIO"] * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"训练集: {train_size}个样本, 验证集: {val_size}个样本")
    
    # 创建模型
    model = FoveaModel(backbone_name=CONFIG["BACKBONE"]).to(CONFIG["DEVICE"])
    
    # 损失函数
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.BCELoss()
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["LEARNING_RATE"],
        weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG["EPOCHS"],
        eta_min=CONFIG["LEARNING_RATE"] * 0.01
    )
    
    # 早停
    best_val_loss = float("inf")
    patience_counter = 0
    
    # 训练历史
    history = {
        "train_loss": [], "val_loss": [],
        "train_reg_loss": [], "train_cls_loss": [],
        "val_reg_loss": [], "val_cls_loss": []
    }
    
    logger.info(f"开始训练（设备：{CONFIG['DEVICE']}）")
    
    for epoch in range(CONFIG["EPOCHS"]):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_reg_loss = 0.0
        train_cls_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} (Train)")
        for batch_idx, batch in enumerate(train_bar):
            images = batch["image"].to(CONFIG["DEVICE"])
            coords = batch["coords"].to(CONFIG["DEVICE"])
            visible = batch["visible"].to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            
            # 前向传播
            reg_out, cls_out = model(images)
            
            # 计算损失
            reg_loss = reg_criterion(reg_out, coords)
            cls_loss = cls_criterion(cls_out, visible)
            
            # 组合损失（固定权重）
            total_loss = reg_loss + 0.5 * cls_loss
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 记录损失
            batch_size = images.size(0)
            train_loss += total_loss.item() * batch_size
            train_reg_loss += reg_loss.item() * batch_size
            train_cls_loss += cls_loss.item() * batch_size
            
            train_bar.set_postfix({
                "loss": total_loss.item(),
                "reg": reg_loss.item(),
                "cls": cls_loss.item()
            })
        
        # 计算平均训练损失
        avg_train_loss = train_loss / train_size
        avg_train_reg_loss = train_reg_loss / train_size
        avg_train_cls_loss = train_cls_loss / train_size
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_reg_loss = 0.0
        val_cls_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} (Val)")
            for batch in val_bar:
                images = batch["image"].to(CONFIG["DEVICE"])
                coords = batch["coords"].to(CONFIG["DEVICE"])
                visible = batch["visible"].to(CONFIG["DEVICE"])
                
                reg_out, cls_out = model(images)
                
                reg_loss = reg_criterion(reg_out, coords)
                cls_loss = cls_criterion(cls_out, visible)
                total_loss = reg_loss + 0.5 * cls_loss
                
                batch_size = images.size(0)
                val_loss += total_loss.item() * batch_size
                val_reg_loss += reg_loss.item() * batch_size
                val_cls_loss += cls_loss.item() * batch_size
                
                val_bar.set_postfix({
                    "loss": total_loss.item(),
                    "reg": reg_loss.item(),
                    "cls": cls_loss.item()
                })
        
        # 计算平均验证损失
        avg_val_loss = val_loss / val_size
        avg_val_reg_loss = val_reg_loss / val_size
        avg_val_cls_loss = val_cls_loss / val_size
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_reg_loss"].append(avg_train_reg_loss)
        history["train_cls_loss"].append(avg_train_cls_loss)
        history["val_reg_loss"].append(avg_val_reg_loss)
        history["val_cls_loss"].append(avg_val_cls_loss)
        
        # 日志输出
        logger.info(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {avg_train_loss:.4f} (Reg: {avg_train_reg_loss:.4f}, Cls: {avg_train_cls_loss:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} (Reg: {avg_val_reg_loss:.4f}, Cls: {avg_val_cls_loss:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "history": history
            }, CONFIG["CHECKPOINT_PATH"])
            logger.info(f"保存最佳模型（Val Loss: {best_val_loss:.4f}）")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["PATIENCE"]:
                logger.info(f"早停触发！最佳Val Loss: {best_val_loss:.4f}")
                break
    
    logger.info("训练完成！")
    return model, history

# ---------------------- 5. 预测函数 ----------------------
def predict_test_set(model):
    logger.info("开始预测测试集...")
    
    # 创建测试集（无数据增强）
    test_dataset = FoveaDataset(
        img_dir=CONFIG["TEST_IMG_DIR"],
        csv_path=CONFIG["TRAIN_CSV_PATH"],  # 这里只是借用路径，不会用到标签
        image_size=CONFIG["IMAGE_SIZE"],
        is_train=False
    )
    
    # 查找测试图片
    test_img_paths = {}
    for img_file in os.listdir(CONFIG["TEST_IMG_DIR"]):
        if any(img_file.endswith(suffix) for suffix in CONFIG["IMAGE_SUFFIXES"]):
            try:
                img_num = int(os.path.splitext(img_file)[0])
                test_img_paths[img_num] = os.path.join(CONFIG["TEST_IMG_DIR"], img_file)
            except ValueError:
                continue
    
    logger.info(f"测试集图片数：{len(test_img_paths)}")
    
    model.eval()
    
    # 按模板顺序预测
    template_order = list(range(81, 101))
    submission_data = []
    
    for img_num in tqdm(template_order, desc="预测测试集"):
        if img_num not in test_img_paths:
            logger.warning(f"图片{img_num}不在测试集中，填充0")
            submission_data.append({"ImageID": f"{img_num}_Fovea_X", "value": 0.00})
            submission_data.append({"ImageID": f"{img_num}_Fovea_Y", "value": 0.00})
            continue
        
        img_path = test_img_paths[img_num]
        
        try:
            # 预处理图像
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size
            
            # 缩放
            scale_x = CONFIG["IMAGE_SIZE"] / orig_w
            scale_y = CONFIG["IMAGE_SIZE"] / orig_h
            image = image.resize((CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"]), Image.BILINEAR)
            
            # 转换为Tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(CONFIG["DEVICE"])
            
            # 预测
            with torch.no_grad():
                reg_out, cls_out = model(image_tensor)
            
            # 后处理
            pred_coords_norm = reg_out.cpu().numpy()[0]
            pred_visible = cls_out.cpu().numpy()[0][0] > 0.5
            
            if pred_visible:
                # 反归一化并缩放到原始尺寸
                pred_x = pred_coords_norm[0] * CONFIG["IMAGE_SIZE"] / scale_x
                pred_y = pred_coords_norm[1] * CONFIG["IMAGE_SIZE"] / scale_y
                
                # 边界检查
                pred_x = max(0, min(orig_w - 1, pred_x))
                pred_y = max(0, min(orig_h - 1, pred_y))
            else:
                pred_x, pred_y = 0.0, 0.0
            
            submission_data.append({
                "ImageID": f"{img_num}_Fovea_X",
                "value": round(float(pred_x), 2)
            })
            submission_data.append({
                "ImageID": f"{img_num}_Fovea_Y",
                "value": round(float(pred_y), 2)
            })
            
        except Exception as e:
            logger.error(f"图片{img_num}预测失败：{str(e)}")
            submission_data.append({"ImageID": f"{img_num}_Fovea_X", "value": 0.00})
            submission_data.append({"ImageID": f"{img_num}_Fovea_Y", "value": 0.00})
    
    # 保存提交文件
    submission_df = pd.DataFrame(submission_data)
    csv_path = os.path.join(CONFIG["OUTPUT_DIR"], "submission.csv")
    submission_df.to_csv(csv_path, index=False)
    
    # 统计信息
    visible_count = sum(1 for i in range(0, len(submission_data), 2) 
                       if submission_data[i]["value"] > 0 or submission_data[i+1]["value"] > 0)
    
    logger.info(f"提交文件已保存：{csv_path}")
    logger.info(f"预测结果统计：共{len(submission_data)//2}张图片，{visible_count}张可见")
    logger.info(f"前10条记录：\n{submission_df.head(20).to_string(index=False)}")
    
    return submission_df

# ---------------------- 6. 主函数 ----------------------
if __name__ == "__main__":
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info("=" * 50)
    logger.info("中心凹检测定位系统")
    logger.info(f"配置：{CONFIG}")
    logger.info("=" * 50)
    
    try:
        # 训练模型
        model, history = train_model()
        
        # 加载最佳模型
        if os.path.exists(CONFIG["CHECKPOINT_PATH"]):
            checkpoint = torch.load(CONFIG["CHECKPOINT_PATH"], map_location=CONFIG["DEVICE"])
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"加载最佳模型（Epoch {checkpoint['epoch']+1}, Val Loss: {checkpoint['val_loss']:.4f})")
        else:
            logger.warning("未找到检查点文件，使用最后训练的模型")
        
        # 预测测试集
        submission_df = predict_test_set(model)
        
        logger.info("? 流程全部完成！")
        logger.info(f"?? 提交文件：{os.path.join(CONFIG['OUTPUT_DIR'], 'submission.csv')}")
        
    except Exception as e:
        logger.error(f"? 流程报错：{str(e)}", exc_info=True)
        raise e