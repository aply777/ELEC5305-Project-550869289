import os, random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ============== Setting ==============
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

H = {
    "speaker_split_ratio": 0.8,
    "batch_size": 64,
    "epochs": 10,
    "lr": 1e-5,
    "dropout": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "root_train": r"E:\2025S2\ELEC5305\TIMIT\TRAIN",
    "root_test":  r"E:\2025S2\ELEC5305\TIMIT\TEST",
    "save_dir":   r"E:\2025S2\ELEC5305\plots_alexnet"
}
os.makedirs(H["save_dir"], exist_ok=True)
device = H["device"]
print("Device:", device)

# ============== transforms ==============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============== Dataset 1:Speaker  ==============
class SpeakerDataset(Dataset):
    def __init__(self, root_dir, split="train", ratio=0.8, transform=None, seed=42):
        self.root = root_dir
        self.transform = transform
        self.samples = []
        self.speakers = sorted([
            d for dr in os.listdir(root_dir) if dr.startswith("DR")
            for d in os.listdir(os.path.join(root_dir, dr))
            if os.path.isdir(os.path.join(root_dir, dr, d))
        ])
        self.spk_to_id = {spk: i for i, spk in enumerate(self.speakers)}

        local_rng = random.Random(seed)

        for dr in sorted(os.listdir(root_dir)):
            dr_path = os.path.join(root_dir, dr)
            if not os.path.isdir(dr_path): continue

            for spk in sorted(os.listdir(dr_path)):
                spk_path = os.path.join(dr_path, spk)
                if not os.path.isdir(spk_path): continue

                files = [f for f in os.listdir(spk_path)
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                files = sorted(files)
                local_rng.shuffle(files)

                cut = int(len(files) * ratio)
                train_files = files[:cut]
                val_files   = files[cut:]
                target_files = train_files if split=="train" else val_files

                for fname in target_files:
                    self.samples.append((
                        os.path.join(spk_path, fname),
                        self.spk_to_id[spk]
                    ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, spk_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(spk_id, dtype=torch.long)

# ============== Dataset 2：Gender（original TRAIN/TEST） ==============
class GenderDataset(Dataset):
    def __init__(self, train_dir, test_dir, split, transform=None):
        self.transform = transform
        self.samples = []
        root = train_dir if split=="train" else test_dir

        for dr in sorted(os.listdir(root)):
            dr_path = os.path.join(root, dr)
            if not os.path.isdir(dr_path): continue

            for spk in sorted(os.listdir(dr_path)):
                spk_path = os.path.join(dr_path, spk)
                if not os.path.isdir(spk_path): continue

                gender = 0 if spk.startswith("F") else 1
                files = [f for f in os.listdir(spk_path)
                         if f.lower().endswith((".png",".jpg",".jpeg"))]

                for fname in files:
                    self.samples.append((
                        os.path.join(spk_path, fname),
                        gender
                    ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, gender = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(gender, dtype=torch.long)

# ============== Dataset 3: Dialect Original TRAIN/TEST ==============
class DialectDataset(Dataset):
    def __init__(self, train_dir, test_dir, split, transform=None):
        self.transform = transform
        self.samples = []
        root = train_dir if split=="train" else test_dir

        for dr in sorted(os.listdir(root)):
            dr_path = os.path.join(root, dr)
            if not os.path.isdir(dr_path): continue
            region_id = int(dr[-1]) - 1  # DR1..DR8 -> 0..7

            for spk in sorted(os.listdir(dr_path)):
                spk_path = os.path.join(dr_path, spk)
                if not os.path.isdir(spk_path): continue

                files = [f for f in os.listdir(spk_path)
                         if f.lower().endswith((".png",".jpg",".jpeg"))]

                for fname in files:
                    self.samples.append((
                        os.path.join(spk_path, fname),
                        region_id
                    ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, reg = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(reg, dtype=torch.long)

# ==============  AlexNet ==============
class MultiTaskAlexNet(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        base = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool = base.avgpool
        self.flatten = nn.Flatten()
        self.shared = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc_spk = nn.Linear(4096, num_speakers)
        self.fc_gen = nn.Linear(4096, 2)
        self.fc_reg = nn.Linear(4096, 8)

    def extract_shared(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.shared(x)
        return x

    def forward(self, x):
        z = self.extract_shared(x)
        return {
            "spk": self.fc_spk(z),
            "gen": self.fc_gen(z),
            "reg": self.fc_reg(z)
        }

# ==============  ==============
def train_one_task(task, num_classes, train_loader, val_loader, model: MultiTaskAlexNet):
    head_map = {
        "SpeakerID": ("fc_spk",  "spk"),
        "Gender":    ("fc_gen",  "gen"),
        "Dialect":   ("fc_reg",  "reg"),
    }
    head_attr, out_key = head_map[task]


    params = list(model.shared.parameters()) + list(getattr(model, head_attr).parameters())
    opt = torch.optim.Adam(params, lr=H["lr"])
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_preds, best_labels = None, None
    hist_train, hist_val = [], []

    for ep in range(1, H["epochs"]+1):
        model.train()
        correct = total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = getattr(model, head_attr)(model.extract_shared(X))
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)
        train_acc = 100 * correct / total if total else 0
        hist_train.append(train_acc)

        # ---- val ----
        model.eval()
        correct = total = 0
        preds, labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = getattr(model, head_attr)(model.extract_shared(X))
                p = logits.argmax(1)
                preds.extend(p.cpu().numpy()); labels.extend(y.cpu().numpy())
                correct += (p == y).sum().item()
                total   += y.size(0)
        val_acc = 100 * correct / total if total else 0
        hist_val.append(val_acc)

        print(f"[{task}] Epoch {ep}/{H['epochs']} | Train={train_acc:.2f}% | Val={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_preds = preds[:]
            best_labels = labels[:]

            torch.save(model.state_dict(), os.path.join(H["save_dir"], f"best_{task}.pth"))
            print(f"  ✅ Saved best {task} (Val {val_acc:.2f}%)")

    # ---- confusion matrix ----
    cm = confusion_matrix(best_labels, best_preds, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(f"{task} – Confusion Matrix")
    plt.savefig(os.path.join(H["save_dir"], f"{task}_cm.png")); plt.close()

    # ---- curve ----
    plt.figure(figsize=(8,5))
    plt.plot(hist_train, label="Train Acc")
    plt.plot(hist_val,   label="Val Acc")
    plt.title(f"{task} Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.grid()
    plt.savefig(os.path.join(H["save_dir"], f"{task}_acc.png")); plt.close()

    return best_acc


print("📌 Loading datasets...")
# Speaker 8:2
train_spk = SpeakerDataset(H["root_train"], split="train", ratio=H["speaker_split_ratio"], transform=transform, seed=SEED)
val_spk   = SpeakerDataset(H["root_train"], split="val",   ratio=H["speaker_split_ratio"], transform=transform, seed=SEED)

# Gender/Dialect: original TRAIN > train TEST > val
train_gen = GenderDataset(H["root_train"], H["root_test"], split="train", transform=transform)
val_gen   = GenderDataset(H["root_train"], H["root_test"], split="val",   transform=transform)

train_reg = DialectDataset(H["root_train"], H["root_test"], split="train", transform=transform)
val_reg   = DialectDataset(H["root_train"], H["root_test"], split="val",   transform=transform)

print(f"Speaker: Train={len(train_spk)} | Val={len(val_spk)} | #Speakers={len(train_spk.spk_to_id)}")
print(f"Gender : Train={len(train_gen)} | Val={len(val_gen)}")
print(f"Dialect: Train={len(train_reg)} | Val={len(val_reg)}")


model = MultiTaskAlexNet(num_speakers=len(train_spk.spk_to_id)).to(device)

print("\n====== TRAIN SPEAKER ID ======")
best_spk = train_one_task(
    "SpeakerID",
    num_classes=len(train_spk.spk_to_id),
    train_loader=DataLoader(train_spk, batch_size=H["batch_size"], shuffle=True),
    val_loader=DataLoader(val_spk, batch_size=H["batch_size"], shuffle=False),
    model=model
)

print("\n====== TRAIN GENDER ======")
best_gen = train_one_task(
    "Gender",
    num_classes=2,
    train_loader=DataLoader(train_gen, batch_size=H["batch_size"], shuffle=True),
    val_loader=DataLoader(val_gen, batch_size=H["batch_size"], shuffle=False),
    model=model
)

print("\n====== TRAIN DIALECT ======")
best_reg = train_one_task(
    "Dialect",
    num_classes=8,
    train_loader=DataLoader(train_reg, batch_size=H["batch_size"], shuffle=True),
    val_loader=DataLoader(val_reg, batch_size=H["batch_size"], shuffle=False),
    model=model
)

print(f"\nAll Tasks Completed!  Best Val Acc — Speaker: {best_spk:.2f}%, Gender: {best_gen:.2f}%, Dialect: {best_reg:.2f}%")
print(f"Plots & CMs saved to: {H['save_dir']}")
