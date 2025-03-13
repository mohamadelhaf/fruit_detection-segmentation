import torch
from torch.utils.data import Dataset, DataLoader
import glob
import yaml
from PIL import Image
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import transforms
import os
import random
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = r'C:\Users\mhdel\runs\detect\train18\weights\best.pt'
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file not found: {weights_path}")


model = YOLO(weights_path).to(device)
train_data_path = r'C:\Users\mhdel\OneDrive\Bureau\proj_fruits\fruit_detection-segmentation\proj_yolo\yolo\Datasets\train\train'
val_data_path = r'C:\Users\mhdel\OneDrive\Bureau\proj_fruits\fruit_detection-segmentation\proj_yolo\yolo\Datasets\test\test'
DATASET_CONFIG = r'C:\Users\mhdel\OneDrive\Bureau\proj_fruits\fruit_detection-segmentation\proj_yolo\yolo\Datasets\dataset.yaml'
with open(DATASET_CONFIG, 'r') as f:
    data_config = yaml.safe_load(f)
num_classes = data_config['nc']
class_names = data_config['names']

transform = transforms.Compose([
    transforms.Resize((640, 640)),  
    transforms.ToTensor(),          
])


class YoloDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        if not self.root.endswith(os.sep):
            self.root += os.sep
        self.img_paths = sorted(glob.glob(self.root + '*.[jp][pn]g'))  
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img) if self.transform else img
            return img, img_path
        except:
            return None, img_path

train_ds = YoloDataset(train_data_path, transform=transform)
val_ds = YoloDataset(val_data_path, transform=transform)
train_dl = DataLoader(train_ds, batch_size=6, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
train_loss_list = []
val_loss_list = []
precision_list = []



def train_yolo(epochs=50, batch_size=6, img_size=640, weights='yolov5s.pt'):
    model = YOLO(weights).to(device)
    #scaler = GradScaler()
    #optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

    results = model.train(
        data=DATASET_CONFIG, 
        epochs=epochs, 
        batch=batch_size,
        imgsz=img_size,
        augment=True,  
        cos_lr=True    
    )
    
    

def get_latest_train_run():
    """Finds the latest YOLOv5 training folder in 'runs/detect/' dynamically."""
    base_dir = r"C:\Users\mhdel\runs\detect"
    train_runs = [d for d in os.listdir(base_dir) if d.startswith("train") and os.path.isdir(os.path.join(base_dir, d))]
    train_runs = [d for d in train_runs if re.match(r"train\d+$", d)]

    if not train_runs:
        print("Error: No valid YOLOv5 training runs found in 'runs/detect/'.")
        return None
    
    full_paths = [os.path.join(base_dir, d) for d in train_runs]
    latest_run = sorted(full_paths, key=lambda x: int(x.split("train")[1]), reverse=True)[0]

    return latest_run

def plot_yolo_metrics():
    latest_run = get_latest_train_run()
    base_dir = r"C:\Users\mhdel\runs\detect"
    if not latest_run:
        return

    results_file = os.path.join(base_dir, latest_run, "results.csv")
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Make sure training has completed.")
        return

    df = pd.read_csv(results_file)
    epochs = df.index 
    train_loss = df["train/box_loss"]
    val_loss = df["val/box_loss"]
    precision = df["metrics/precision(B)"]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss", linestyle="-", marker="o", color="blue")
    plt.plot(epochs, val_loss, label="Validation Loss", linestyle="-", marker="o", color="red")
    plt.plot(epochs, precision, label="Precision", linestyle="-", marker="o", color="green")

    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title(f"Training Metrics ({latest_run})")
    plt.legend()
    plt.grid()
    plt.show()




def load_ground_truth(label_path, img_width, img_height, resized_size=(640, 640)):
    boxes = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height
                x_min = x_center - (width / 2)
                y_min = y_center - (height / 2)
                x_max = x_center + (width / 2)
                y_max = y_center + (height / 2)
                x_min = x_min * (resized_size[0] / img_width)
                y_min = y_min * (resized_size[1] / img_height)
                x_max = x_max * (resized_size[0] / img_width)
                y_max = y_max * (resized_size[1] / img_height)
                boxes.append((x_min, y_min, x_max, y_max, class_id))
    except FileNotFoundError:
        pass  
    return boxes


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

@torch.no_grad()
def validate_image(img_path, model):
    img_name = os.path.basename(img_path)
    label_path = os.path.join(val_data_path, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    pil_image = Image.open(img_path).convert('RGB')
    img_width, img_height = pil_image.size
    gt_boxes = load_ground_truth(label_path, img_width, img_height)
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    results = model(img_tensor, conf=0.5)
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img_np)
    for gt_box in gt_boxes:
        x_min, y_min, x_max, y_max, class_id = gt_box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min, f'GT: {class_names[class_id]}', 
                bbox=dict(facecolor='blue', alpha=0.5))

    for r in results:
        for box in r.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            pred_label = model.names[int(box.cls)]
            score = box.conf.item()
            iou_scores = [compute_iou((x_min, y_min, x_max, y_max), gt_box[:4]) for gt_box in gt_boxes]
            best_iou = max(iou_scores) if iou_scores else 0
            if best_iou > 0.6:
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                        linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min, f'{pred_label}: {score:.2f}, IoU: {best_iou:.2f}', 
                        bbox=dict(facecolor='yellow', alpha=0.5))
    plt.title(f'Predictions for {img_name}')
    plt.show()

@torch.no_grad()
def validate_loop(val_folder, model, num_images=5):
    image_files = [f for f in os.listdir(val_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    for img_name in selected_images:
        img_path = os.path.join(val_folder, img_name)
        validate_image(img_path, model)

def validate_model(model, val_dl):
    model.eval()
    val_loss = 0
    total_preds = 0
    correct_preds = 0
    iou_threshold = 0.5
    all_iou_scores = []
    criterion = torch.nn.CrossEntropyLoss()  
    with torch.no_grad():
        for images, img_paths in val_dl:
            images = images.to(device)
            outputs = model(images)
            
            batch_loss = 0
            batch_ious = []
            for i, img_path in enumerate(img_paths):
                label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt')
                gt_boxes = load_ground_truth(label_path, img_width=640, img_height=640)

                if not gt_boxes:
                    continue
                predictions = outputs[i].boxes if hasattr(outputs[i], 'boxes') else []

                if predictions:
                    pred_boxes = predictions.xyxy.cpu().numpy()
                    pred_classes = predictions.cls.cpu().numpy()
                    pred_confs = predictions.conf.cpu().numpy()
                else:
                    pred_boxes = []
                    pred_classes = []
                    pred_confs = []

                if len(pred_boxes) == 0:
                    continue
                gt_tensors = torch.tensor([box[:4] for box in gt_boxes]).to(device)
                gt_classes = torch.tensor([box[4] for box in gt_boxes]).to(device)
                loss = criterion(torch.tensor(pred_classes).to(device), gt_classes)
                batch_loss += loss.item()
                for pred_box in pred_boxes:
                    iou_scores = [compute_iou(pred_box, gt_box[:4]) for gt_box in gt_boxes]
                    best_iou = max(iou_scores) if iou_scores else 0
                    batch_ious.append(best_iou)
    
                    if best_iou > iou_threshold:
                        correct_preds += 1
                    total_preds += 1
                
            val_loss += batch_loss / len(img_paths) if len(img_paths) > 0 else 0
            if batch_ious:
                all_iou_scores.extend(batch_ious)
    
    avg_val_loss = val_loss / len(val_dl) if len(val_dl) > 0 else 0
    precision = correct_preds / total_preds if total_preds > 0 else 0
    map_value = sum(all_iou_scores) / len(all_iou_scores) if all_iou_scores else 0
    
    return avg_val_loss, precision, map_value


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support() 
    print('Script started...')
    train_yolo(epochs=100, batch_size=8, img_size=640, weights='yolov5s.pt')
    validate_loop(val_data_path, model, num_images=5)
    plot_yolo_metrics()

