import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import pandas as pd
import os
import logging
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

CONFIG = {
    "TRAIN_IMG_DIR": "/kaggle/input/project4project4/detection/train",
    "TRAIN_CSV_PATH": "/kaggle/input/project4project4/detection/fovea_localization_train_GT.csv",
    "TEST_IMG_DIR": "/kaggle/input/project4project4/detection/test",
    "OUTPUT_DIR": "/kaggle/working",
    "BATCH_SIZE": 8,
    "EPOCHS": 15,
    "LEARNING_RATE": 0.0008,
    "WEIGHT_DECAY": 1e-4,
    "VAL_RATIO": 0.2,
    "PATIENCE": 4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BACKBONE": "resnet18",
    "RESUME_TRAIN": False,
    "CHECKPOINT_PATH": "/kaggle/working/best_model.pth",
    "REG_LOSS_WEIGHT": 1.0,
    "CLS_LOSS_WEIGHT": 0.5,
    "IMAGE_SUFFIXES": [".jpg", ".png", ".jpeg"]
}

# ---------------------- 2. 日志配置 ----------------------
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

# ---------------------- 3. 数据集类（图片名匹配+异常过滤） ----------------------
class FoveaDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None, is_train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.image_suffixes = CONFIG["IMAGE_SUFFIXES"]
        
        try:

            self.train_img_map = {}
            for img_file in os.listdir(img_dir):
                if any(img_file.endswith(suffix) for suffix in self.image_suffixes):
                    img_base = os.path.splitext(img_file)[0]  # 如"0081"
                    self.train_img_map[img_base] = os.path.join(img_dir, img_file)
            
            logger.info(f"训练集图片数：{len(self.train_img_map)}，基名示例：{list(self.train_img_map.keys())[:5]}")
            

            self.raw_df = pd.read_csv(csv_path)
            self.df = self.raw_df.copy()
            self.df["data"] = self.df["data"].astype(int).astype(str).str.zfill(4)  # 1→"0001"
            logger.info(f"CSV格式化后前5个data值：{self.df['data'].head(5).values}")
            
            # 过滤有效样本
            self.df = self.df[self.df["data"].isin(self.train_img_map.keys())]
            self.df = self.df[(self.df["Fovea_X"] >= 0) & (self.df["Fovea_Y"] >= 0)]
            if "is_visible" not in self.df.columns:
                self.df["is_visible"] = 1.0
            
            logger.info(f"有效样本数：{len(self.df)}（原始{len(self.raw_df)}个）")
            if len(self.df) == 0:
                raise ValueError("无有效样本！请检查图片基名或CSV数据")
            
        except Exception as e:
            logger.error(f"加载失败：{str(e)}")
            raise e

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_name_base = row["data"].strip()
            img_path = self.train_img_map[img_name_base]
            img_name_full = os.path.basename(img_path)
            
            image_ori = Image.open(img_path).convert("RGB")
            ori_w, ori_h = image_ori.size
            
            fovea_x = row["Fovea_X"]
            fovea_y = row["Fovea_Y"]
            fovea_x_norm = fovea_x / ori_w if ori_w > 0 else 0.0
            fovea_y_norm = fovea_y / ori_h if ori_h > 0 else 0.0
            
            if self.transform:
                image = self.transform(image_ori)
            else:
                image = transforms.ToTensor()(image_ori)
            
            if self.is_train:
                coords = torch.tensor([fovea_x_norm, fovea_y_norm], dtype=torch.float32)
                visible_label = torch.tensor([row["is_visible"]], dtype=torch.float32)
                return image, coords, visible_label
            else:
                return image, img_name_full, (ori_w, ori_h)
        
        except Exception as e:
            logger.warning(f"样本{idx}处理失败：{str(e)}，跳过")
            return torch.zeros(3, 224, 224), torch.zeros(2), torch.zeros(1)

# ---------------------- 4. 数据增强 ----------------------
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=3, expand=False),
        transforms.ColorJitter(brightness=0.03, contrast=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

# ---------------------- 5. 模型定义 ----------------------
class FoveaModel(nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super().__init__()
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"仅支持resnet18，当前输入：{backbone_name}")
        
        for param in list(self.backbone.parameters())[:-8]:
            param.requires_grad = False
        
        self.backbone.fc = nn.Identity()
        self.reg_head = nn.Linear(num_features, 2)
        self.cls_head = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        reg_out = self.reg_head(features)
        cls_out = self.cls_head(features)
        return reg_out, cls_out

# ---------------------- 6. 早停类 ----------------------
class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
        
def train_model():
    train_transform, val_test_transform = get_transforms()
    full_dataset = FoveaDataset(
        img_dir=CONFIG["TRAIN_IMG_DIR"],
        csv_path=CONFIG["TRAIN_CSV_PATH"],
        transform=train_transform
    )
    
    val_size = max(1, int(CONFIG["VAL_RATIO"] * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    if train_size <= 0:
        raise ValueError(f"训练集为空！请减小VAL_RATIO（当前{CONFIG['VAL_RATIO']}）")
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_test_transform
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=0)
    
    model = FoveaModel(backbone_name=CONFIG["BACKBONE"]).to(CONFIG["DEVICE"])
    reg_criterion = nn.SmoothL1Loss()
    cls_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["LEARNING_RATE"],
        weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    

    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=CONFIG["PATIENCE"])
    
    start_epoch = 0
    if CONFIG["RESUME_TRAIN"] and os.path.exists(CONFIG["CHECKPOINT_PATH"]):
        checkpoint = torch.load(CONFIG["CHECKPOINT_PATH"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"断点续训：从第{start_epoch}轮开始")
    
    logger.info(f"开始训练（设备：{CONFIG['DEVICE']}，训练集{train_size}个样本）")
    best_val_loss = float("inf")
    for epoch in range(start_epoch, CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} (Train)")
        for images, coords, visible_labels in train_bar:
            images = images.to(CONFIG["DEVICE"])
            coords = coords.to(CONFIG["DEVICE"])
            visible_labels = visible_labels.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            reg_out, cls_out = model(images)
            r_loss = reg_criterion(reg_out, coords)
            c_loss = cls_criterion(cls_out, visible_labels)
            total_loss = CONFIG["REG_LOSS_WEIGHT"] * r_loss + CONFIG["CLS_LOSS_WEIGHT"] * c_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * images.size(0)
            train_bar.set_postfix({"loss": total_loss.item()})
        
        avg_train_loss = train_loss / train_size
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} (Val)")
            for images, coords, visible_labels in val_bar:
                images = images.to(CONFIG["DEVICE"])
                coords = coords.to(CONFIG["DEVICE"])
                visible_labels = visible_labels.to(CONFIG["DEVICE"])
                
                reg_out, cls_out = model(images)
                r_loss = reg_criterion(reg_out, coords)
                c_loss = cls_criterion(cls_out, visible_labels)
                total_loss = CONFIG["REG_LOSS_WEIGHT"] * r_loss + CONFIG["CLS_LOSS_WEIGHT"] * c_loss
                
                val_loss += total_loss.item() * images.size(0)
                val_bar.set_postfix({"loss": total_loss.item()})
        
        avg_val_loss = val_loss / val_size
        scheduler.step(avg_val_loss)
        
        logger.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss
            }, CONFIG["CHECKPOINT_PATH"])
            logger.info(f"保存最优模型（Val Loss: {best_val_loss:.4f}）")
        
        if early_stopping(avg_val_loss):
            logger.info("早停触发，停止训练")
            break
    
    logger.info("训练完成！")
    return model


def predict_test_set(model):
    test_transform = get_transforms()[1]

    template_order = list(range(81, 101))  # [81,82,...,100]
 
    img_path_map = {}
    for img_file in os.listdir(CONFIG["TEST_IMG_DIR"]):
        if img_file.endswith(tuple(CONFIG["IMAGE_SUFFIXES"])):
            img_num = int(os.path.splitext(img_file)[0])  # 0081→81
            img_path_map[img_num] = os.path.join(CONFIG["TEST_IMG_DIR"], img_file)
    
    logger.info(f"测试集图片映射完成，共{len(img_path_map)}张图片")
    if len(img_path_map) != 20:
        logger.warning(f"测试集图片数不为20（当前{len(img_path_map)}张），可能影响提交完整性")
    

    model.eval()
    submission_data = []
    test_bar = tqdm(template_order, desc="按模板顺序预测")
    with torch.no_grad():
        for img_num in test_bar:
            try:
                img_path = img_path_map[img_num]
                img_name_full = os.path.basename(img_path)
                
                # 加载图像
                image_ori = Image.open(img_path).convert("RGB")
                ori_w, ori_h = image_ori.size
                image = test_transform(image_ori).unsqueeze(0).to(CONFIG["DEVICE"])
                
                # 预测
                reg_out, cls_out = model(image)
                is_visible = torch.sigmoid(cls_out).cpu().numpy()[0][0] > 0.5
                pred_x_norm, pred_y_norm = reg_out.cpu().numpy()[0]
                pred_x = pred_x_norm * ori_w if is_visible else 0.0
                pred_y = pred_y_norm * ori_h if is_visible else 0.0
                
             
                submission_data.append({
                    "ImageID": f"{img_num}_Fovea_X",
                    "value": round(pred_x, 2)
                })
                submission_data.append({
                    "ImageID": f"{img_num}_Fovea_Y",
                    "value": round(pred_y, 2)
                })
            
            except Exception as e:
                logger.warning(f"图片{img_num}（{img_name_full}）预测失败：{str(e)}，填充0.00")
         
                submission_data.append({"ImageID": f"{img_num}_Fovea_X", "value": 0.00})
                submission_data.append({"ImageID": f"{img_num}_Fovea_Y", "value": 0.00})
    
    # 4. 保存提交文件
    submission_df = pd.DataFrame(submission_data)
    csv_path = os.path.join(CONFIG["OUTPUT_DIR"], "submission.csv")
    submission_df.to_csv(csv_path, index=False)
    logger.info(f"提交文件生成完成！路径：{csv_path}，共{len(submission_data)}条记录")
    logger.info(f"前10条预览：\n{submission_df.head(10).to_string(index=False)}")


if __name__ == "__main__":
    try:
        model = train_model()
        # 加载最优模型预测
        best_checkpoint = torch.load(CONFIG["CHECKPOINT_PATH"])
        model.load_state_dict(best_checkpoint["model_state_dict"])
        logger.info(f"加载最优模型（训练轮次：{best_checkpoint['epoch']+1}）")
        predict_test_set(model)
        logger.info("✅ 全部流程完成！可在/kaggle/working目录下载submission.csv直接提交")
    except Exception as e:
        logger.error(f"❌ 流程报错：{str(e)}", exc_info=True)
        raise e