import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import glob
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.ops as ops


train_root = r"C:\Users\mhdel\OneDrive\Bureau\elhaf_tp_obj_detect\elhaf_tp_obj_detect\Datasets\Datasets\train"
test_root  = r"C:\Users\mhdel\OneDrive\Bureau\elhaf_tp_obj_detect\elhaf_tp_obj_detect\Datasets\Datasets\test"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = {'background': 0, 'apple': 1, 'banana': 2, 'orange': 3}
labels2targets = {l: t for t, l in enumerate(labels)}
targets2labels = {t: l for l, t in labels2targets.items()}
num_classes = len(targets2labels)


map_values = []
train_losses = []
print("Script started...")
sys.stdout.flush()

class FruitsDataset(Dataset):
    def __init__(self, root, transforms=None): 
        self.root = root 
        self.transforms = transforms 
        self.img_paths = sorted(glob.glob(self.root + '/*.jpg')) 
        self.xml_paths = sorted(glob.glob(self.root + '/*.xml'))  
        self.valid_data = [
    (img_path, img_path.rsplit('.jpg', 1)[0] + '.xml') 
    for img_path in self.img_paths 
    if img_path.rsplit('.jpg', 1)[0] + '.xml' in self.xml_paths]

    def __len__(self): 
        return len(self.valid_data)

    def __getitem__(self, idx):
        img_path, xml_path = self.valid_data[idx]
        img = Image.open(img_path).convert('RGB')
        orig_width, orig_height = img.size
        scale = 224.0 / max(orig_width, orig_height)
        new_width, new_height = int(orig_width * scale), int(orig_height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        padded_img = Image.new('RGB', (224, 224), (0, 0, 0))
        pad_x, pad_y = (224 - new_width) // 2, (224 - new_height) // 2
        padded_img.paste(img, (pad_x, pad_y))
        xml = ET.parse(xml_path)
        objects = xml.findall('object')
        boxes = []
        labels_list = []

        for obj in objects:
            XMin = float(obj.find('bndbox').find('xmin').text)
            YMin = float(obj.find('bndbox').find('ymin').text)
            XMax = float(obj.find('bndbox').find('xmax').text)
            YMax = float(obj.find('bndbox').find('ymax').text)
            scaled_XMin = (XMin * scale) + pad_x
            scaled_YMin = (YMin * scale) + pad_y
            scaled_XMax = (XMax * scale) + pad_x
            scaled_YMax = (YMax * scale) + pad_y

            box = [scaled_XMin, scaled_YMin, scaled_XMax, scaled_YMax]
            boxes.append(box)
            labels_list.append(labels2targets[obj.find('name').text])
        target = {
            'labels': torch.tensor(labels_list, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float32)
        }
        if self.transforms:
            padded_img = self.transforms(padded_img)

        return padded_img, target

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


tr_ds = FruitsDataset(train_root, transforms=transform)
val_ds = FruitsDataset(test_root, transforms=transform)
tr_dl = DataLoader(tr_ds, batch_size=6, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

def get_model(num_classes): 
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

model = get_model(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
#use for sgd optimizer to monitor the learning rate

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def mean_average_precision(predictions, targets, iou_threshold=0.5):
    all_precisions = []
    all_recalls = []

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        gt_boxes = target['boxes'].cpu().numpy()
        
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue
        score_threshold = 0.6
        mask = pred_scores > score_threshold
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        if len(pred_boxes) == 0:
            continue

        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        assigned_gt = []

        for i, pb in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1

            for j, gb in enumerate(gt_boxes):
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx not in assigned_gt:
                tp[i] = 1
                assigned_gt.append(best_gt_idx)
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / (len(gt_boxes) + 1e-6)
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)
        all_precisions.append(np.mean(precision))
        all_recalls.append(np.mean(recall))

    return np.mean(all_precisions) if all_precisions else 0, np.mean(all_recalls) if all_recalls else 0




@torch.no_grad()
def validate(val_dl, model):
    model.eval()
    all_predictions = []
    all_targets = []
    all_images = []

    for images, targets in val_dl:
        images = [img.to(device) for img in images]
        outputs = model(images)

        all_predictions.extend(outputs)
        all_targets.extend(targets)
        all_images.extend(images)

    return all_images, all_predictions, all_targets

def train_batch(batch, model, optimizer): 
    model.train() 
    imgs, targets = batch
    imgs = [img.to(device) for img in imgs]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(imgs, targets)
    loss = sum(loss_value for loss_value in losses.values())
    loss.backward()
    optimizer.step()
    return loss.item()




n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    train_loss_sum = 0.0
    num_train_batches = len(tr_dl)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} starting with LR = {current_lr:.2e}", flush=True)

    for i, batch in enumerate(tr_dl):
        loss = train_batch(batch, model, optimizer)
        train_loss_sum += loss
        print(f"Epoch {epoch+1} Batch {i+1}/{num_train_batches} Loss: {loss:.4f}", flush=True)

    avg_loss = train_loss_sum / num_train_batches
    train_losses.append(avg_loss)  
    print(f"Epoch {epoch+1} Completed: Average Train Loss: {avg_loss:.4f}", flush=True)

    all_images, all_predictions, all_targets = validate(val_dl, model)
    mAP, recall = mean_average_precision(all_predictions, all_targets)
    map_values.append(mAP)  
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    print(f"Mean Recall: {recall:.4f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, n_epochs + 1), map_values, marker='o', linestyle='-', label='mAP')
plt.xlabel("Epochs")
plt.ylabel("Mean Average Precision (mAP)")
plt.title("mAP over Training Epochs")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs + 1), train_losses, marker='o', linestyle='-', color='orange', label='Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


def denormalize(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean  
    return np.clip(img, 0, 1)

score_threshold = 0.6 #min0.6
iou_threshold = 0.3 #to make around 0.3-0.4


def normalize_bbox(x_min, y_min, x_max, y_max, img_shape):
    x_min = max(0, x_min / 224 * img_shape[1])
    y_min = max(0, y_min / 224 * img_shape[0])
    x_max = min(img_shape[1], x_max / 224 * img_shape[1])
    y_max = min(img_shape[0], y_max / 224 * img_shape[0])
    return x_min, y_min, x_max, y_max





def visualize_results(all_images, all_predictions, all_targets):
    for i in range(min(10, len(all_images))):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        img_numpy = all_images[i].cpu().permute(1, 2, 0).numpy()
        img_numpy = denormalize(img_numpy)
        ax.imshow(img_numpy)

        output = all_predictions[i]
        target = all_targets[i]
        pred_boxes = output['boxes'].cpu().numpy()
        pred_labels = output['labels'].cpu().numpy()
        pred_scores = output['scores'].cpu().numpy()
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()

        print(f"predictions for image {i}: {output}", flush=True)
        print(f" before thresholding: {len(pred_boxes)} predicted boxes", flush=True)

    
        mask = pred_scores > score_threshold
        pred_boxes, pred_labels, pred_scores = pred_boxes[mask], pred_labels[mask], pred_scores[mask]
        if len(pred_boxes) > 0:
            keep_indices = ops.nms(torch.tensor(pred_boxes), torch.tensor(pred_scores), iou_threshold=iou_threshold)
            keep_indices = keep_indices.cpu().numpy()  
            pred_boxes = [pred_boxes[i] for i in keep_indices]
            pred_labels = [pred_labels[i] for i in keep_indices]
            pred_scores = [pred_scores[i] for i in keep_indices]

        print(f"after NMS: {len(pred_boxes)} predicted boxes remain", flush=True)
        for box, label in zip(gt_boxes, gt_labels):
            x_min, y_min, x_max, y_max = box
            x_min, y_min, x_max, y_max = normalize_bbox(x_min, y_min, x_max, y_max, img_numpy.shape)
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, f"GT: {targets2labels[label]}", 
                    bbox=dict(facecolor='blue', alpha=0.5))

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x_min, y_min, x_max, y_max = box
            x_min, y_min, x_max, y_max = normalize_bbox(x_min, y_min, x_max, y_max, img_numpy.shape)
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, f"{targets2labels[label]}: {score:.2f}", 
                    bbox=dict(facecolor='yellow', alpha=0.5))

            
            ious = [compute_iou(box, gt_box) for gt_box in gt_boxes]
            best_iou = max(ious) if ious else 0
            precision = best_iou
            print(f"Precision for bbox {box}: {precision:.4f}")

        plt.show()




all_images, all_predictions, all_targets = validate(val_dl, model)
mAP, recall = mean_average_precision(all_predictions, all_targets)
map_values.append(mAP)
print(f"Mean Average Precision (mAP): {mAP:.4f}")
print(f"Mean Recall: {recall:.4f}")
visualize_results(all_images, all_predictions, all_targets)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(map_values) + 1), map_values, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Mean Average Precision (mAP)")
plt.title("mAP over Training Epochs")
plt.grid(True)
plt.show()