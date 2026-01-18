import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import subprocess


BASE = "/kaggle/input/555555/segmentation"

TRAIN_IMG_DIR  = f"{BASE}/train/image"
TRAIN_MASK_DIR = f"{BASE}/train/label"
TEST_IMG_DIR   = f"{BASE}/test/image"
CSV_SCRIPT     = f"{BASE}/segmentation_to_csv.py"

WORK_DIR = "/kaggle/working"
OUT_IMG_DIR = f"{WORK_DIR}/image"
os.makedirs(OUT_IMG_DIR, exist_ok=True)

IMG_SIZE = 512
THRESHOLD = 0.45


def apply_clahe(img):
    img = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img) / 255.0

def resize(img, size=IMG_SIZE):
    return cv2.resize(img, (size, size))

def random_flip_rotate(img, mask):
    # 翻转
    if np.random.rand() > 0.5:
        img, mask = np.flip(img, 1), np.flip(mask, 1)
    if np.random.rand() > 0.5:
        img, mask = np.flip(img, 0), np.flip(mask, 0)
    # 随机旋转
    k = np.random.randint(0, 4)
    img, mask = np.rot90(img, k), np.rot90(mask, k)
    return img, mask


class VesselDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, train=False, img_size=IMG_SIZE):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.train = train
        self.names = sorted(os.listdir(img_dir))
        self.img_size = img_size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))[:, :, 1] / 255.0
        img = apply_clahe(resize(img, self.img_size))

        if self.mask_dir:
            mask = cv2.imread(os.path.join(self.mask_dir, name), 0)
            mask = resize((mask == 0).astype(np.float32), self.img_size)
            if self.train:
                img, mask = random_flip_rotate(img, mask)
            return (
                torch.tensor(img.copy(), dtype=torch.float32).unsqueeze(0),
                torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0)
            )

        return torch.tensor(img.copy(), dtype=torch.float32).unsqueeze(0), name



class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def up(x, ref):
    return F.interpolate(x, ref.shape[2:], mode="bilinear", align_corners=False)

class UNetPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.c00 = ConvBlock(1, 64)
        self.c10 = ConvBlock(64, 128)
        self.c20 = ConvBlock(128, 256)
        self.c30 = ConvBlock(256, 512)

        self.c01 = ConvBlock(64+128, 64)
        self.c11 = ConvBlock(128+256, 128)
        self.c21 = ConvBlock(256+512, 256)

        self.c02 = ConvBlock(64*2+128, 64)
        self.c12 = ConvBlock(128*2+256, 128)

        self.c03 = ConvBlock(64*3+128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x00 = self.c00(x)
        x10 = self.c10(self.pool(x00))
        x20 = self.c20(self.pool(x10))
        x30 = self.c30(self.pool(x20))

        x01 = self.c01(torch.cat([x00, up(x10, x00)], 1))
        x11 = self.c11(torch.cat([x10, up(x20, x10)], 1))
        x21 = self.c21(torch.cat([x20, up(x30, x20)], 1))

        x02 = self.c02(torch.cat([x00, x01, up(x11, x00)], 1))
        x12 = self.c12(torch.cat([x10, x11, up(x21, x10)], 1))

        x03 = self.c03(torch.cat([x00, x01, x02, up(x12, x00)], 1))
        return self.out(x03)


def dice_loss(p, t):
    p = torch.sigmoid(p)
    inter = (p * t).sum()
    return 1 - (2*inter + 1)/(p.sum() + t.sum() + 1)

def focal_loss(p, t, gamma=2):
    bce = F.binary_cross_entropy_with_logits(p, t, reduction="none")
    pt = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()

def total_loss(p, t):
    return 0.4*dice_loss(p,t) + 0.4*F.binary_cross_entropy_with_logits(p,t) + 0.2*focal_loss(p,t)



def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetPP().to(device)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)  # 去掉 verbose

    loader = DataLoader(
        VesselDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train=True),
        batch_size=4, shuffle=True, num_workers=2, pin_memory=True
    )

    best_loss = float("inf")
    for e in range(80):
        model.train()
        loss_sum = 0
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            with autocast():
                loss = total_loss(model(img), mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
        epoch_loss = loss_sum / len(loader)
        print(f"Epoch {e+1}/80 | Loss {epoch_loss:.4f}")
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"{WORK_DIR}/best_model.pth")



def predict():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetPP().to(device)
    model.load_state_dict(torch.load(f"{WORK_DIR}/best_model.pth", map_location=device))
    model.eval()

    loader = DataLoader(VesselDataset(TEST_IMG_DIR), batch_size=1)

    with torch.no_grad():
        for img, name in loader:
            img = img.to(device)
            preds = []

            # 8种TTA翻转
            for flip_h in [False, True]:
                for flip_v in [False, True]:
                    x = img.clone()
                    if flip_h: x = torch.flip(x, [2])
                    if flip_v: x = torch.flip(x, [3])
                    p = torch.sigmoid(model(x))
                    if flip_h: p = torch.flip(p, [2])
                    if flip_v: p = torch.flip(p, [3])
                    preds.append(p)
            pred = torch.mean(torch.stack(preds), 0)[0,0].cpu().numpy()
            mask = 255 - (pred > THRESHOLD).astype(np.uint8)*255
            cv2.imwrite(os.path.join(OUT_IMG_DIR, name[0]), mask)


train()
predict()

subprocess.run(["python", CSV_SCRIPT], check=True)