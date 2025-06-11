import argparse, os, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pandas as pd

FOLDER2LABEL = {
    "Exposed Rebar": 0,
    "X-shape": 3,                 
    "Diagonal": 4,                 
    "Vertical_large": 6,
    "Vertical": 7,
    "Horizontal_large": 8,
    "Horizontal": 9,
    }  

CRITERIA_SUBSET = {
    18: [0, 3, 4, 6, 8],                 
    19: [5, 7],         
    20: [1, 9, 10],       
}

class ColumnCriteriaDataset(Dataset):
    def __init__(self, root_dir: Path, target_class: int, tfms):
        self.paths, self.ys = [], []
        self.valid = CRITERIA_SUBSET[target_class]
        self.l2i = {l: i for i, l in enumerate(self.valid)}
        self.tfms = tfms

        for folder in (root_dir).glob("*"):
            name = folder.name
            if name not in FOLDER2LABEL:
                continue
            crit_id = FOLDER2LABEL[name]

            if   crit_id in (0, 3, 4, 6, 8):          dmg = 18
            elif crit_id in (5, 7):    dmg = 19
            else:                            dmg = 20
            if dmg != target_class:
                continue    

            for img in folder.glob("*"):
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                self.paths.append(img)
                self.ys.append(crit_id)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.tfms: img = self.tfms(img)
        y = torch.zeros(len(self.valid))
        y[self.l2i[self.ys[idx]]] = 1.0
        return img, y

@torch.no_grad()
def evaluate(model, loader, device, th=0.5):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = (logits.sigmoid() > th).float()
        # sample-wise exact match accuracy
        match = (preds == y).all(dim=1).sum().item()
        correct += match
        total += x.size(0)
    return correct / total if total else 0.0

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    root = Path(args.crack_root)
    full_ds = ColumnCriteriaDataset(root, args.target_class, tfm)

    val_len = int(0.1 * len(full_ds)) if len(full_ds) >= 10 else 0
    train_ds, val_ds = (full_ds, None) if val_len == 0 else \
                       random_split(full_ds, [len(full_ds)-val_len, val_len],
                                    generator=torch.Generator().manual_seed(42))

    loader_tr = DataLoader(train_ds, batch_size=args.bs,
                           shuffle=True,  num_workers=4, drop_last=len(train_ds) > args.bs)
    loader_val = None if val_ds is None else \
                 DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4)

    backbone = models.resnet34(weights=None)
    backbone.fc = nn.Identity()
    backbone.load_state_dict(torch.load(args.class_ckpt, map_location="cpu"),
                              strict=False)     
    for p in backbone.parameters(): p.requires_grad = False

    out_dim = len(CRITERIA_SUBSET[args.target_class])
    head = nn.Linear(512, out_dim)
    model = nn.Sequential(backbone, head).to(device)

    opt = torch.optim.Adam(head.parameters(), lr=args.lr)  
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    for ep in range(args.epochs):
        model.train(); running = 0.0
        for x, y in tqdm(loader_tr, leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); opt.step()
            running += loss.item() * x.size(0)
        tr_loss = running / len(train_ds)

        if loader_val:
            val_acc = evaluate(model, loader_val, device, th=args.th)
            print(f"Epoch {ep+1:02d} | loss {tr_loss:.4f} | val-acc {val_acc:.3f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), args.out)
                print(f"  ✓ better ({best_acc:.3f}) → saved {args.out}")
        else:
            print(f"Epoch {ep+1:02d} | loss {tr_loss:.4f}")

    if not loader_val:
        torch.save(model.state_dict(), args.out)
        print("  ✓ saved (no val):", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_class", type=int, choices=[18,19,20], required=True)
    ap.add_argument("--crack_root", default="column_crack")
    ap.add_argument("--class_ckpt", default="column_class.pth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--th", type=float, default=0.4, help="val accuracy threshold")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())