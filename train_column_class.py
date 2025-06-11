import argparse, random, os, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

class ColumnClassDataset(Dataset):
    LABEL_MAP = {"Class A": (18, 0), "Class B": (19, 1), "Class C": (20, 2)}

    def __init__(self, root_dir: Path, tfms=None):
        self.samples = []     # (img_path, class_idx)
        self.tfms = tfms
        for cls_folder in ["Class A", "Class B", "Class C"]:
            img_paths = (root_dir / cls_folder).glob("*")
            for p in img_paths:
                if p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    _, idx = self.LABEL_MAP[cls_folder]
                    self.samples.append((p, idx))

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.tfms: img = self.tfms(img)
        return img, torch.tensor(y, dtype=torch.long)

def run_epoch(model, loader, optim, criterion, device, train=True):
    model.train() if train else model.eval()
    tot, correct, running = 0, 0, 0.
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        if train:
            optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward(); optim.step()
        running += loss.item() * x.size(0)
        tot += x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return running / tot, correct / tot

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf_train = transforms.Compose([
        transforms.Resize(256), transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    tf_val = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])

    dataset = ColumnClassDataset(Path(args.data_root), tf_train)
    val_len = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset)-val_len, val_len],
                                    generator=torch.Generator().manual_seed(SEED))
    val_ds.dataset.tfms = tf_val
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=4)

    model = models.resnet34(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(512, 3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0
    for ep in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, optim, criterion, device, True)
        vl_loss, vl_acc = run_epoch(model, val_loader, optim, criterion, device, False)
        print(f"[{ep+1}/{args.epochs}]  train {tr_loss:.4f}/{tr_acc:.3f} | "
              f"val {vl_loss:.4f}/{vl_acc:.3f}")
        if vl_acc >= best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), args.out)
            print(" ✓ saved:", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="column_damage")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="column_class.pth")
    main(ap.parse_args())
