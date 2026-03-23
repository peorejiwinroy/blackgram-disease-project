import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys

# --- FIX FOR MODULE ERRORS ---
# This adds the 'src' folder to the search path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

# Now import the model from src/model.py
try:
    from model import HybridBlackgramNet
except ImportError:
    st.error("Could not find model.py in the src folder.")

# --- 1. CONFIGURATION ---
DATA_DIR = r"D:\bld2\dataset"
BATCH_SIZE = 16
EPOCHS = 30  
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATA PREPARATION ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_loop(use_quantum, save_name):
    print(f"\n" + "="*50)
    print(f"🔥 STARTING TRAINING: {'QAgroNet' if use_quantum else 'Standard'}")
    print("="*50)

    # Load Data
    train_path = os.path.join(DATA_DIR, 'train')
    if not os.path.exists(train_path):
        print(f"❌ Error: {train_path} not found!")
        return

    train_dataset = datasets.ImageFolder(train_path, train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model Structure
    model = HybridBlackgramNet(num_classes=5, use_quantum=use_quantum).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    # Save only the state_dict (weights) for app.py compatibility
    torch.save(model.state_dict(), save_name)
    print(f"✅ SUCCESSFULLY SAVED: {save_name}")

# --- 3. EXECUTION ---
if __name__ == "__main__":
    # RUN 1: QAgroNet (Hybrid Quantum)
    # This generates model_qagronet.pth
    train_loop(use_quantum=True, save_name="model_qagronet.pth")
    
    # RUN 2: Standard Classical (ResNet50)
    # This generates model_standard.pth
    train_loop(use_quantum=False, save_name="model_standard.pth")
    
    print("\n🎉 ALL TRAINING SESSIONS COMPLETE!")
    print("Check D:\\bld2 for model_qagronet.pth and model_standard.pth")
