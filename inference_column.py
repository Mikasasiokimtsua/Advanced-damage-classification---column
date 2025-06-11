import argparse, csv, glob, os, torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

CRIT_SUB = {18:[0,3,4,6,8], 19:[5,7], 20:[1,9,10]}

def build_criteria_head(weight_path, out_dim, backbone):
    head = nn.Sequential(backbone, nn.Linear(512, out_dim))
    head.load_state_dict(torch.load(weight_path, map_location="cpu"))
    head.eval()
    return head

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cls_model = models.resnet34()
    cls_model.fc = nn.Linear(512,3)
    cls_model.load_state_dict(torch.load(args.class_ckpt, map_location=device))
    cls_model = cls_model.to(device).eval()

    base_backbone = models.resnet34()
    base_backbone.fc = nn.Identity()
    base_backbone.load_state_dict(torch.load(args.class_ckpt, map_location="cpu"),
                                   strict=False)
    for p in base_backbone.parameters(): p.requires_grad=False

    crit_A = build_criteria_head(args.critA_ckpt, 5, base_backbone).to(device)
    crit_B = build_criteria_head(args.critB_ckpt, 2, base_backbone).to(device)
    crit_C = build_criteria_head(args.critC_ckpt, 3, base_backbone).to(device)

    tfms = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor()])

    img_paths = sorted(glob.glob(os.path.join(args.test_dir,"*.jpg")),
                       key=lambda p:int(os.path.splitext(os.path.basename(p))[0]))

    with open(args.out_csv,"w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID","class"])
        for p in tqdm(img_paths):
            img_id = os.path.splitext(os.path.basename(p))[0]
            x = tfms(Image.open(p).convert("RGB")).unsqueeze(0).to(device)

            cls_idx = cls_model(x).argmax(1).item()   
            cls_label = [18,19,20][cls_idx]

            if cls_label == 18:
                logits = crit_A(x).sigmoid()[0]; subset = CRIT_SUB[18]
            elif cls_label == 19:
                logits = crit_B(x).sigmoid()[0]; subset = CRIT_SUB[19]
            else:
                logits = crit_C(x).sigmoid()[0]; subset = CRIT_SUB[20]

            crit = [str(l) for l,s in zip(subset, logits) if s > args.th]
            if not crit:                      
                if cls_label == 20:
                    crit = ["1"]      
                elif cls_label == 18:
                    crit = ["0"]    
            joined = ",".join([str(cls_label)] + crit)      
            f.write(f'{img_id},"{joined}"\n') 

    print("✓ submission saved:", args.out_csv)

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--class_ckpt", default="column_class.pth")
    ap.add_argument("--critA_ckpt",  default="column_criteria_A.pth")
    ap.add_argument("--critB_ckpt",  default="column_criteria_B.pth")
    ap.add_argument("--critC_ckpt",  default="column_criteria_C.pth")
    ap.add_argument("--th", type=float, default=0.4)
    ap.add_argument("--out_csv", default="submission.csv")
    main(ap.parse_args())
