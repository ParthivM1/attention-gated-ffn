import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
import sys

# Import your modules
from models.vit import VisionTransformer
from layers.geodynamic_layer import GeoDynamicLayer

def get_device():
   if torch.cuda.is_available(): return torch.device("cuda")
   if torch.backends.mps.is_available(): return torch.device("mps")
   return torch.device("cpu")

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--batch_size", type=int, default=128)
   parser.add_argument("--dataset", type=str, default="cifar100")
   parser.add_argument("--checkpoint_dir", type=str, default="/root/checkpoints")
   parser.add_argument("--checkpoint_name", type=str, default=None, help="Specific .pth file to load")
   args = parser.parse_args()

   device = get_device()
   print(f"🚀 Evaluating on device: {device}")

   # 1. Data Setup (Test Set)
   # Must match training normalization
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
   ])

   if args.dataset == "cifar100":
       test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
       num_classes = 100
   else:
       test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
       num_classes = 10

   test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

   # 2. Model Init
   # CRITICAL: These must match the optimized training parameters!
   print("🏗️  Initializing Geo-ViT...")
   model = VisionTransformer(
       img_size=32,
       patch_size=4,      # UPDATED from 16 -> 4 (Matches training)
       embed_dim=192,
       depth=6,
       num_heads=6,       # UPDATED from 3 -> 6 (Matches training)
       num_classes=num_classes,
       linear_layer=GeoDynamicLayer
   )
   model.to(device)

   # 3. Load Checkpoint
   if args.checkpoint_name:
       ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
   else:
       # Find the file with the highest epoch number
       print(f"🔍 Searching for checkpoints in {args.checkpoint_dir}...")
       if not os.path.exists(args.checkpoint_dir):
           print("❌ Checkpoint directory not found.")
           return
           
       files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pth")]
       if not files:
           print("❌ No checkpoints found in volume!")
           return
       
       # Sort to find the latest (e.g., geovit_e50.pth)
       # We assume naming format geovit_e{number}.pth
       try:
           files.sort(key=lambda x: int(x.split('_e')[-1].split('.')[0]))
       except:
           files.sort() # Fallback to alphabetical
           
       ckpt_path = os.path.join(args.checkpoint_dir, files[-1])
  
   print(f"📦 Loading Checkpoint: {ckpt_path}")
   checkpoint = torch.load(ckpt_path, map_location=device)
   
   # Handle cases where checkpoint is just state_dict or full dict
   if 'model_state_dict' in checkpoint:
       model.load_state_dict(checkpoint['model_state_dict'])
       epoch = checkpoint.get('epoch', -1)
       train_acc = checkpoint.get('acc', 0.0)
       print(f"   -> Loaded State from Epoch {epoch+1} (Train Acc: {train_acc:.2f}%)")
   else:
       model.load_state_dict(checkpoint)
       epoch = "Unknown"
  
   # 4. Evaluation Loop
   model.eval()
   criterion = nn.CrossEntropyLoss()
   total_loss = 0
   correct = 0
   total = 0
  
   print("🏁 Starting Evaluation...")
   with torch.no_grad():
       for batch_idx, (inputs, targets) in enumerate(test_loader):
           inputs, targets = inputs.to(device), targets.to(device)
           outputs = model(inputs)
           loss = criterion(outputs, targets)

           total_loss += loss.item()
           _, predicted = outputs.max(1)
           total += targets.size(0)
           correct += predicted.eq(targets).sum().item()
          
           if (batch_idx + 1) % 20 == 0:
               print(f"   Batch {batch_idx+1}/{len(test_loader)}: Acc so far: {100.*correct/total:.2f}%")

   acc = 100. * correct / total
   avg_loss = total_loss / len(test_loader)
  
   print("="*40)
   print(f"✅ RESULTS for Checkpoint: {os.path.basename(ckpt_path)}")
   print(f"Test Accuracy: {acc:.2f}%")
   print(f"Test Loss:     {avg_loss:.4f}")
   print("="*40)

if __name__ == "__main__":
   main()