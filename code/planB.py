import os, random, numpy as np, pandas as pd
import torch, torchaudio, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from speechbrain.inference import SpeakerRecognition

# ============== hyperparameter ==============
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

H = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 10,
    "batch_size": 64,
    "lr": 1e-3,
    "dropout": 0.3,
    "sample_rate": 16000,
    "emb_dim": 192,
    "root_train": r"E:\2025S2\ELEC5305\TIMIT\TRAIN",
    "root_test":  r"E:\2025S2\ELEC5305\TIMIT\TEST",
    "output_dir": r"E:\2025S2\ELEC5305\output",
    "speaker_split_ratio": 0.8,  # SpeakerID 8:2
}
os.makedirs(H["output_dir"], exist_ok=True)
device = H["device"]
print(f"Device: {device.upper()}")

# ============== pretrain ECAPA ==============
print("Loading SpeechBrain ECAPA-TDNN ...")
ecapa = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
ecapa.mods.embedding_model.to(device)
ecapa.mods.mean_var_norm.to(device)
ecapa.device = device
ecapa.eval()
print("ECAPA ready")

# ============== label ==============
def get_gender(speaker_id):  # F -> 0, M -> 1
    return 0 if speaker_id[0].upper() == "F" else 1
def get_region(dr_folder):   # DR1..DR8 -> 0..7
    return int(dr_folder[-1]) - 1

# ============== only for *_fixed.wav, split(train/test) ==============
def extract_embeddings_with_native_split(root_dir, native_split_tag):
    rows = []
    for dr in sorted(os.listdir(root_dir)):
        dr_path = os.path.join(root_dir, dr)
        if not os.path.isdir(dr_path): continue
        for spk in sorted(os.listdir(dr_path)):
            spk_path = os.path.join(dr_path, spk)
            if not os.path.isdir(spk_path): continue
            wavs = [f for f in os.listdir(spk_path)
                    if f.lower().endswith(".wav") and "_fixed" in f.lower()]
            for fname in sorted(wavs):
                fpath = os.path.join(spk_path, fname)
                try:
                    wav, sr = torchaudio.load(fpath)
                    if sr != H["sample_rate"]:
                        wav = torchaudio.transforms.Resample(sr, H["sample_rate"])(wav)
                    with torch.no_grad():
                        emb = ecapa.encode_batch(wav.to(device)).squeeze().cpu().numpy()
                    rows.append({
                        "path": fpath,
                        "speaker": spk,
                        "gender": get_gender(spk),
                        "region": get_region(dr),
                        "native_split": native_split_tag,  # "train" or "test"
                        **{f"e{i}": emb[i] for i in range(H["emb_dim"])}
                    })
                except Exception as e:
                    print("skip:", fpath, "|", e)
    return pd.DataFrame(rows)

print("Extracting embeddings (TRAIN native)...")
df_train_native = extract_embeddings_with_native_split(H["root_train"], "train")
print("Extracting embeddings (TEST native)...")
df_test_native  = extract_embeddings_with_native_split(H["root_test"],  "test")

df_all = pd.concat([df_train_native, df_test_native], ignore_index=True)
print(f"Embeddings done. All={len(df_all)}, TRAIN(native)={len(df_train_native)}, TEST(native)={len(df_test_native)}")


df_train_native.to_csv(os.path.join(H["output_dir"], "timit_train_native_emb.csv"), index=False)
df_test_native.to_csv(os.path.join(H["output_dir"], "timit_test_native_emb.csv"), index=False)

# ============== SpeakerID: everyspeaker 8:2  TRAIN+TEST > df_all ==============
def split_speaker_internal(df_all, ratio=0.8):
    flags = []
    for spk, sub in df_all.groupby("speaker"):
        idx = list(sub.index)
        random.shuffle(idx)
        cut = int(len(idx) * ratio)
        spk_train = set(idx[:cut])
        for i in sub.index:
            flags.append(("train" if i in spk_train else "val"))
    # groupby
    df_all = df_all.copy()
    df_all["speaker_split"] = flags
    df_spk_train = df_all[df_all["speaker_split"] == "train"].reset_index(drop=True)
    df_spk_val   = df_all[df_all["speaker_split"] == "val"].reset_index(drop=True)
    return df_spk_train, df_spk_val

df_spk_train, df_spk_val = split_speaker_internal(df_all, H["speaker_split_ratio"])
print(f"SpeakerID split: train={len(df_spk_train)}, val={len(df_spk_val)}")


le_spk = LabelEncoder().fit(df_all["speaker"])
df_spk_train["speaker_id"] = le_spk.transform(df_spk_train["speaker"])
df_spk_val["speaker_id"]   = le_spk.transform(df_spk_val["speaker"])
num_spk_classes = len(le_spk.classes_)

# ============== Gender / Dialect:(TRAIN > train, TEST > val) ==============
df_gender_train = df_train_native.reset_index(drop=True).copy()
df_gender_val   = df_test_native.reset_index(drop=True).copy()
df_dial_train   = df_train_native.reset_index(drop=True).copy()
df_dial_val     = df_test_native.reset_index(drop=True).copy()

# ============== Dataset / Model  ==============
class EmbDataset(Dataset):
    def __init__(self, df, target):
        emb_cols = [f"e{i}" for i in range(H["emb_dim"])]
        self.X = torch.tensor(df[emb_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target].values, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, out_dim)
        )
    def forward(self, x): return self.net(x)

def train_task(task_name, num_classes, df_tr, df_va, label_col):
    model = Classifier(H["emb_dim"], num_classes, H["dropout"]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=H["lr"])
    crit  = nn.CrossEntropyLoss()

    tr_loader = DataLoader(EmbDataset(df_tr, label_col), batch_size=H["batch_size"], shuffle=True)
    va_loader = DataLoader(EmbDataset(df_va, label_col), batch_size=H["batch_size"], shuffle=False)

    best_val, best_true, best_pred = 0.0, None, None
    hist = {"train_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(1, H["epochs"]+1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for X, y in tr_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = crit(out, y)
            loss.backward(); opt.step()
            tot_loss += loss.item()
            correct  += (out.argmax(1) == y).sum().item()
            total    += y.size(0)
        train_acc = 100*correct/total if total else 0

        # val
        model.eval()
        v_correct, v_total = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in va_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                pred = out.argmax(1)
                y_true.extend(y.cpu().numpy()); y_pred.extend(pred.cpu().numpy())
                v_correct += (pred == y).sum().item()
                v_total   += y.size(0)
        val_acc = 100*v_correct/v_total if v_total else 0

        hist["train_loss"].append(tot_loss)
        hist["train_acc"].append(train_acc)
        hist["val_acc"].append(val_acc)
        print(f"[{task_name}] Epoch {ep}/{H['epochs']} | Loss={tot_loss:.2f} | Train={train_acc:.2f}% | Val={val_acc:.2f}%")

        if val_acc > best_val:
            best_val, best_true, best_pred = val_acc, y_true, y_pred
            torch.save(model.state_dict(), os.path.join(H["output_dir"], f"{task_name}.pth"))
            print(f"  ✅ Saved best {task_name} (Val {val_acc:.2f}%)")

    # confusion matrix
    cm = confusion_matrix(best_true, best_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"{task_name} – Confusion Matrix (best)")
    plt.savefig(os.path.join(H["output_dir"], f"{task_name}_cm.png")); plt.close()

    # curve
    plt.figure(); plt.plot(hist["train_acc"], label="Train Acc"); plt.plot(hist["val_acc"], label="Val Acc"); plt.legend()
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title(f"{task_name} Accuracy"); 
    plt.savefig(os.path.join(H["output_dir"], f"{task_name}_acc.png")); plt.close()

    plt.figure(); plt.plot(hist["train_loss"], label="Train Loss"); plt.legend()
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{task_name} Loss")
    plt.savefig(os.path.join(H["output_dir"], f"{task_name}_loss.png")); plt.close()

    print(f" {task_name} done. Best Val Acc = {best_val:.2f}%\n")

# ============== 3 task ==============
# SpeakerID every speaker 8:2   speaker_id 
train_task("SpeakerID", num_spk_classes, df_spk_train, df_spk_val, "speaker_id")

# Gender
train_task("Gender", 2, df_gender_train, df_gender_val, "gender")

# Dialect
train_task("Dialect", 8, df_dial_train, df_dial_val, "region")

print("All tasks complete!")
