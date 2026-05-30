import os
import sys
import urllib.request
import zipfile
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Dataset Downloading and Parsing
# ==========================================
DATASET_URL = "https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/plaid_nilm_submetered_dataset.zip"
ZIP_PATH = "plaid_nilm_submetered_dataset.zip"
EXTRACT_DIR = "plaid_dataset"

def download_and_extract():
    is_valid = False
    if os.path.exists(ZIP_PATH):
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as z:
                if z.testzip() is None:
                    is_valid = True
        except Exception:
            pass
            
    if not is_valid:
        print(f"Downloading dataset from {DATASET_URL}...")
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        if os.path.exists(EXTRACT_DIR):
            import shutil
            shutil.rmtree(EXTRACT_DIR)
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print("Download completed.")
    else:
        print("Dataset zip exists and is valid. Skipping download.")

    if not os.path.exists(EXTRACT_DIR):
        print(f"Extracting dataset to {EXTRACT_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction completed.")
    else:
        print("Dataset already extracted. Skipping extraction.")

# ==========================================
# 2. Feature Extraction Pipeline (FFT + LogDB)
# ==========================================
def extract_features_for_file(filepath):
    """
    Apply TI's exact preprocessing pipeline:
    Hanning Window -> RFFT (32pt) -> Positive Half (17 bins) -> Absolute ->
    Binning (17 to 8 bins with skip DC and size 2) -> Log10 -> 8 Frame Concat.
    """
    df = pd.read_csv(filepath, skiprows=1, header=None)
    data = df.values[:1600, :].astype(np.float32)  # shape (1600, 2)
    data = data.T
    
    frame_size = 32
    num_frames = 1600 // frame_size  # 50 frames
    num_frame_concat = 8
    
    hanning = np.hanning(frame_size)
    
    channel_features = []
    for ch in range(2):
        ch_data = data[ch]
        ch_binned = []
        for f in range(num_frames):
            frame = ch_data[f * frame_size : (f + 1) * frame_size]
            win_frame = frame * hanning
            fft_res = np.fft.fft(win_frame)
            fft_half = fft_res[:17]
            mag = np.abs(fft_half) / 32.0
            
            binned = []
            for b in range(8):
                bin_val = np.sum(mag[1 + b*2 : 1 + b*2 + 2])
                binned.append(bin_val)
            binned = np.array(binned)
            
            log_db = 20.0 * np.log10(binned + 1e-100)
            ch_binned.append(log_db)
            
        ch_binned = np.array(ch_binned)  # shape (50, 8)
        
        concat_features = []
        for i in range(num_frame_concat - 1, num_frames):
            frame_block = ch_binned[i - num_frame_concat + 1 : i + 1]  # shape (8, 8)
            concat_features.append(frame_block.flatten())
            
        channel_features.append(concat_features)
        
    channel_features = np.array(channel_features)  # shape (2, 43, 64)
    channel_features = channel_features.transpose(1, 0, 2)
    channel_features = np.expand_dims(channel_features, axis=-1)
    
    return channel_features  # shape (43, 2, 64, 1)

# ==========================================
# 3. Custom PyTorch Dataset
# ==========================================
class NILMDataset(Dataset):
    def __init__(self, list_file, dataset_dir):
        self.samples = []
        self.labels = []
        
        self.class_names = sorted([
            "air_conditioner", "compact_fluorescent_lamp", "fan",
            "fridge", "hair_dryer", "heater", "incandescent_light_bulb",
            "laptop", "microwave", "vacuum", "washing_machine"
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        with open(list_file, 'r') as f:
            lines = f.readlines()
            
        print(f"Loading features for list: {list_file}...")
        for line in lines:
            rel_path = line.strip()
            if not rel_path:
                continue
            class_name = rel_path.split('/')[0]
            norm_class_name = class_name.lower().replace(" ", "_")
            if norm_class_name == "hairdryer":
                norm_class_name = "hair_dryer"
            label_idx = self.class_to_idx[norm_class_name]
            
            filepath = os.path.join(dataset_dir, "classes", rel_path)
            feats = extract_features_for_file(filepath)  # shape (43, 2, 64, 1)
            
            for f in range(feats.shape[0]):
                self.samples.append(feats[f])
                self.labels.append(label_idx)
                
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"Loaded {self.samples.shape[0]} samples with shape {self.samples.shape[1:]}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]), self.labels[idx]

class DummyNILMDataset(Dataset):
    def __init__(self, num_samples):
        self.samples = np.random.randn(num_samples, 2, 64, 1).astype(np.float32)
        self.labels = np.random.randint(0, 11, size=(num_samples,)).astype(np.int64)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]), self.labels[idx]

# ==========================================
# 4. PyTorch Model Definition
# ==========================================
class CLS13k_NPU(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5,1), stride=(2,1), padding=(2,0))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0))
        self.bn6 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn0(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# ==========================================
# 5. Training Loop
# ==========================================
def train_model():
    print("\n==========================================")
    print("Starting Model Training...")
    print("==========================================")
    
    dataset_dir = "plaid_dataset"
    if "--dummy" in sys.argv or not os.path.exists(os.path.join(dataset_dir, "classes")):
        print("Using DUMMY data for training...")
        train_dataset = DummyNILMDataset(500)
        val_dataset = DummyNILMDataset(100)
    else:
        annotations_dir = os.path.join(dataset_dir, "annotations")
        if not os.path.exists(annotations_dir) or not os.path.exists(os.path.join(annotations_dir, "training_list.txt")):
            os.makedirs(annotations_dir, exist_ok=True)
            train_list_path = os.path.join(annotations_dir, "training_list.txt")
            val_list_path = os.path.join(annotations_dir, "validation_list.txt")
            
            classes_dir = os.path.join(dataset_dir, "classes")
            import random
            random.seed(42)
            
            train_lines = []
            val_lines = []
            
            for class_folder in sorted(os.listdir(classes_dir)):
                class_path = os.path.join(classes_dir, class_folder)
                if os.path.isdir(class_path):
                    csv_files = [f for f in sorted(os.listdir(class_path)) if f.endswith(".csv")]
                    random.shuffle(csv_files)
                    split_idx = int(0.8 * len(csv_files))
                    for f in csv_files[:split_idx]:
                        train_lines.append(f"{class_folder}/{f}")
                    for f in csv_files[split_idx:]:
                        val_lines.append(f"{class_folder}/{f}")
                        
            with open(train_list_path, "w") as f:
                f.write("\n".join(train_lines) + "\n")
            with open(val_list_path, "w") as f:
                f.write("\n".join(val_lines) + "\n")
            print(f"Generated {len(train_lines)} train and {len(val_lines)} val annotations.")

        train_dataset = NILMDataset(os.path.join(dataset_dir, "annotations/training_list.txt"), dataset_dir)
        val_dataset = NILMDataset(os.path.join(dataset_dir, "annotations/validation_list.txt"), dataset_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = CLS13k_NPU().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.04, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    best_acc = 0.0
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for samples, targets in train_loader:
            samples, targets = samples.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * samples.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        train_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for samples, targets in val_loader:
                samples, targets = samples.to(device), targets.to(device)
                outputs = model(samples)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1}/30 - Loss: {train_loss/total:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model updated.")
            
    print(f"Training complete. Best Validation Accuracy: {best_acc:.2f}%")

# ==========================================
# 6. Testing / Evaluation Function
# ==========================================
def test_model():
    print("\n==========================================")
    print("Evaluating Model on Test/Validation Set...")
    print("==========================================")
    
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found! Please train the model first using 'python train.py --train'.")
        return
        
    dataset_dir = "plaid_dataset"
    if "--dummy" in sys.argv or not os.path.exists(os.path.join(dataset_dir, "classes")):
        print("Using DUMMY data for evaluation...")
        test_dataset = DummyNILMDataset(200)
    else:
        test_list_path = os.path.join(dataset_dir, "annotations/validation_list.txt")
        test_dataset = NILMDataset(test_list_path, dataset_dir)
        
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLS13k_NPU()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for samples, targets in test_loader:
            samples = samples.to(device)
            outputs = model(samples)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate global metrics
    correct = (all_preds == all_targets).sum()
    total = len(all_targets)
    accuracy = 100.0 * correct / total
    print(f"\nGlobal Evaluation Results:")
    print(f"  Total Test Samples: {total}")
    print(f"  Correctly Classified: {correct}")
    print(f"  Global Accuracy: {accuracy:.2f}%")
    
    # Class-wise performance metrics
    class_names = sorted([
        "air_conditioner", "compact_fluorescent_lamp", "fan",
        "fridge", "hair_dryer", "heater", "incandescent_light_bulb",
        "laptop", "microwave", "vacuum", "washing_machine"
    ])
    
    print("\nClass-wise Performance Metrics:")
    print(f"{'Appliance Class':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Samples':<8}")
    print("-" * 75)
    
    for idx, class_name in enumerate(class_names):
        tp = np.sum((all_preds == idx) & (all_targets == idx))
        fp = np.sum((all_preds == idx) & (all_targets != idx))
        fn = np.sum((all_preds != idx) & (all_targets == idx))
        support = np.sum(all_targets == idx)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{class_name:<30} | {precision:10.2%} | {recall:10.2%} | {f1_score:10.2%} | {support:<8}")
        
    print("-" * 75)

if __name__ == "__main__":
    # Check if dataset needs downloading
    if "--dummy" not in sys.argv:
        download_and_extract()
        
    # Read CLI args
    if "--test" in sys.argv:
        test_model()
    elif "--train" in sys.argv:
        train_model()
    else:
        # Default behavior: Train and then immediately test
        train_model()
        test_model()
