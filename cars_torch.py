#!/usr/bin/env python
"""
car_torch.py – training script for Algerian‑Cars brand classifier
===============================================================
Run:
    python car_torch.py --data ./dataset/DATA

Extras in this revision
-----------------------
* **Live metric tracking** – per‑epoch train/val loss & accuracy stored in CSV (`metrics.csv`).
* **Training curves** – auto‑saves `train_curves.png` (loss + accuracy).
* **Verbose console logs** – prints loss, accuracy, LR and epoch timing, plus `tqdm` minibatch bars.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.utils import save_image

# -------------------------------------------------------------
# 0. Reproducibility
# -------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------------------------------------
# 1. CLI
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Train car brand classifier (MobileNetV2)")
    p.add_argument("--data", default="./dataset/DATA", help="root directory with class folders")
    p.add_argument("--img-size", type=int, default=288)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)            # head only
    p.add_argument("--fine-epochs", type=int, default=20)       # full fine‑tune
    p.add_argument("--init-lr", type=float, default=1e-3)
    p.add_argument("--fine-lr", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--save-preproc", type=int, default=0)
    p.add_argument("--metrics-file", default="metrics.csv")
    return p.parse_args()

# -------------------------------------------------------------
# 2. Dataset helpers
# -------------------------------------------------------------

def collect_paths(root: Path):
    paths, labels, class_to_idx = [], [], {}
    for idx, cls in enumerate(sorted(d.name for d in root.iterdir() if d.is_dir())):
        class_to_idx[cls] = idx
        cls_dir = root / cls
        for img_path in itertools.chain(cls_dir.glob("*.jpg"), cls_dir.glob("*.jpeg"), cls_dir.glob("*.png")):
            paths.append(str(img_path))
            labels.append(idx)
    return np.array(paths), np.array(labels), class_to_idx

class ImageDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.p = paths; self.y = labels; self.tf = tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB")
        return self.tf(img), self.y[i]

# -------------------------------------------------------------
# 3. Transforms
# -------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(sz):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(sz, scale=(0.5,1.0), ratio=(0.75,1.333)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4,0.4,0.4,0.2),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize()
        ], p=0.8),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2)
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(sz*1.15)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return train_tf, eval_tf

# Debug dump ----------------------------------------------------

def save_preproc(loader, out_dir: Path, max_items=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    denorm = transforms.Normalize(mean=[-m/s for m,s in zip(IMAGENET_MEAN, IMAGENET_STD)], std=[1/s for s in IMAGENET_STD])
    written = 0
    for x,_ in loader:
        for i in range(x.size(0)):
            tensor = x[i]
            torch.save(tensor, out_dir/f"{written:04d}.pt")
            save_image(denorm(tensor).clamp(0,1), out_dir/f"{written:04d}.png")
            written += 1
            if written >= max_items: return

# -------------------------------------------------------------
# 4. Model
# -------------------------------------------------------------

def build_model(num_classes, device):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m.to(device)

# -------------------------------------------------------------
# 5. Train / Eval loops
# -------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None, device=torch.device("cpu")):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    loop = tqdm(loader, leave=False)
    with torch.set_grad_enabled(train_mode):
        for x,y in loop:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out,y)
            if train_mode:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()*x.size(0)
            correct  += (out.argmax(1)==y).sum().item()
            total    += x.size(0)
            loop.set_description("train" if train_mode else "eval ")
            loop.set_postfix(loss=loss_sum/total, acc=correct/total)
    return loss_sum/total, correct/total

# -------------------------------------------------------------
# 6. Metric CSV & plotting
# -------------------------------------------------------------

def init_csv(path):
    with path.open("w",newline="") as f:
        csv.writer(f).writerow(["epoch","phase","loss","accuracy"])

def append_csv(path, epoch, phase, loss, acc):
    with path.open("a",newline="") as f:
        csv.writer(f).writerow([epoch,phase,f"{loss:.6f}",f"{acc:.6f}"])

def plot_curves(tr_losses,val_losses,tr_accs,val_accs,out="train_curves.png"):
    ep = range(1,len(tr_losses)+1)
    fig,ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(ep,tr_losses,label="train loss"); ax1.plot(ep,val_losses,label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax2=ax1.twinx(); ax2.plot(ep,tr_accs,"--",label="train acc"); ax2.plot(ep,val_accs,"--",label="val acc")
    ax2.set_ylabel("accuracy")
    h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2,l1+l2,loc="upper left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"Saved training curves → {out}")

# -------------------------------------------------------------
# 7. Main
# -------------------------------------------------------------

def main():
    args = parse_args()
    root = Path(args.data); assert root.exists(), f"dataset root {root} not found"

    metrics_path = Path(args.metrics_file); init_csv(metrics_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split
    paths,labels,class2idx = collect_paths(root)
    print(f"Images: {len(paths)}, classes: {len(class2idx)}")
    sss1 = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=SEED)
    train_idx,tmp_idx = next(sss1.split(paths,labels))
    paths_train,y_train = paths[train_idx], labels[train_idx]
    paths_tmp,y_tmp = paths[tmp_idx], labels[tmp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1,test_size=0.5,random_state=SEED)
    val_idx,test_idx = next(sss2.split(paths_tmp,y_tmp))
    paths_val,y_val = paths_tmp[val_idx], y_tmp[val_idx]
    paths_test,y_test = paths_tmp[test_idx], y_tmp[test_idx]
    print(f"Split → train {len(paths_train)}, val {len(paths_val)}, test {len(paths_test)}")

    tf_train,tf_eval = get_transforms(args.img_size)
    train_ld = DataLoader(ImageDS(paths_train,y_train,tf_train),batch_size=args.batch,shuffle=True,num_workers=2)
    val_ld   = DataLoader(ImageDS(paths_val,y_val,tf_eval),batch_size=args.batch,shuffle=False,num_workers=2)
    test_ld  = DataLoader(ImageDS(paths_test,y_test,tf_eval),batch_size=args.batch,shuffle=False,num_workers=2)

    if args.save_preproc:
        save_preproc(train_ld, Path("_preproc_debug"), max_items=args.save_preproc)
        print(f"[debug] Saved {args.save_preproc} pre-processed images → _preproc_debug/")

    model = build_model(len(class2idx),device)
    criterion = nn.CrossEntropyLoss()

    tr_losses,tr_accs,val_losses,val_accs = [],[],[],[]

    # Stage 1 ----------------------------------------------------
    opt = optim.Adam(filter(lambda p:p.requires_grad,model.parameters()), lr=args.init_lr)
    sch = CosineAnnealingLR(opt,T_max=args.epochs)
    best_val, patience = 0.0, args.patience
    for epoch in range(1,args.epochs+1):
        t0=time.time()
        tr_loss,tr_acc = run_epoch(model,train_ld,criterion,opt,device)
        val_loss,val_acc = run_epoch(model,val_ld,criterion,None,device)
        sch.step()
        tr_losses.append(tr_loss); tr_accs.append(tr_acc)
        val_losses.append(val_loss); val_accs.append(val_acc)
        append_csv(metrics_path,epoch,"train",tr_loss,tr_acc)
        append_csv(metrics_path,epoch,"val",val_loss,val_acc)
        dt=time.time()-t0; lr_now = sch.get_last_lr()[0]
        print(f"[Head {epoch:02d}/{args.epochs}] loss {tr_loss:.4f}/{val_loss:.4f} acc {tr_acc:.3f}/{val_acc:.3f} lr {lr_now:.2e} ({dt:.1f}s)")
        if val_acc>best_val:
            best_val=val_acc; patience=args.patience
            torch.save(model.state_dict(),"best_head.pth")
        else:
            patience-=1
            if patience==0:
                print("Early stop (head phase)"); break

    # Stage 2 ----------------------------------------------------
    for p in model.features.parameters(): p.requires_grad=True
    opt = optim.Adam(model.parameters(), lr=args.fine_lr)
    sch = CosineAnnealingLR(opt,T_max=args.fine_epochs)
    best_val_ft, patience = best_val, args.patience
    for epoch in range(1,args.fine_epochs+1):
        ep_global = len(tr_losses)+1
        t0=time.time()
        tr_loss,tr_acc = run_epoch(model,train_ld,criterion,opt,device)
        val_loss,val_acc = run_epoch(model,val_ld,criterion,None,device)
        sch.step()
        tr_losses.append(tr_loss); tr_accs.append(tr_acc)
        val_losses.append(val_loss); val_accs.append(val_acc)
        append_csv(metrics_path,ep_global,"train",tr_loss,tr_acc)
        append_csv(metrics_path,ep_global,"val",val_loss,val_acc)
        dt=time.time()-t0; lr_now = sch.get_last_lr()[0]
        print(f"[FT   {epoch:02d}/{args.fine_epochs}] loss {tr_loss:.4f}/{val_loss:.4f} acc {tr_acc:.3f}/{val_acc:.3f} lr {lr_now:.2e} ({dt:.1f}s)")
        if val_acc>best_val_ft:
            best_val_ft=val_acc; patience=args.patience
            torch.save(model.state_dict(),"best_full.pth")
        else:
            patience-=1
            if patience==0:
                print("Early stop (fine-tune)"); break

    # Test -------------------------------------------------------
    best_ckpt = "best_full.pth" if Path("best_full.pth").exists() else "best_head.pth"
    model.load_state_dict(torch.load(best_ckpt,map_location=device))
    test_loss,test_acc = run_epoch(model,test_ld,criterion,None,device)
    append_csv(metrics_path,0,"test",test_loss,test_acc)
    print(f"TEST loss {test_loss:.4f} acc {test_acc:.3f}")

    # Curves & save --------------------------------------------
    plot_curves(tr_losses,val_losses,tr_accs,val_accs)
    torch.save(model.state_dict(),"cars_brands.pth")
    print("Final weights saved → cars_brands.pth")


if __name__ == "__main__":
    main()
