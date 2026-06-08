"""
CS2 round prediction training script
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import timm

from dataset import CS2RoundDataset, default_transform
from models import CS2Predictor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="direction", choices=["direction", "winner"])
    parser.add_argument("--mode", type=str, default="single", choices=["single", "sequence"])
    parser.add_argument("--model", type=str, default="swin_tiny", choices=["swin_tiny", "convnext_tiny"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", type=str, default="data/dataset_by_match")
    parser.add_argument("--plant-times", type=str, default="data/plant_times.json")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def create_dataloaders(args):
    train_ds = CS2RoundDataset(
        args.dataset_dir,
        args.plant_times,
        split="train",
        task=args.task,
        max_frames=args.max_frames,
        mode=args.mode,
        img_size=args.img_size,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
    
    val_ds = CS2RoundDataset(
        args.dataset_dir,
        args.plant_times,
        split="val",
        task=args.task,
        max_frames=args.max_frames,
        mode=args.mode,
        img_size=args.img_size,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def create_model(args):
    num_classes = 3 if args.task == "direction" else 2
    
    if args.mode == "single":
        model = CS2Predictor(
            num_classes=num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size,
            dropout=args.dropout,
        )
    else:
        from models import CS2SequencePredictor
        model = CS2SequencePredictor(
            num_classes=num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size,
            dropout=args.dropout,
        )
    
    return model


def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).float().sum()
    return correct / targets.size(0)


def evaluate(model, loader, criterion, device, task):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # split by has_plant for winner task
    correct_has_plant = 0
    correct_no_plant = 0
    total_has_plant = 0
    total_no_plant = 0
    
    with torch.no_grad():
        for batch in loader:
            if task == "winner":
                inputs, labels, has_plant = batch
                inputs, labels, has_plant = inputs.to(device), labels.to(device), has_plant.to(device)
            else:
                inputs, labels, _ = batch
                inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).float()
            
            total_correct += correct.sum().item()
            total_samples += inputs.size(0)
            
            if task == "winner":
                mask_has = has_plant == 1
                mask_no = has_plant == 0
                
                if mask_has.any():
                    correct_has_plant += correct[mask_has].sum().item()
                    total_has_plant += mask_has.sum().item()
                
                if mask_no.any():
                    correct_no_plant += correct[mask_no].sum().item()
                    total_no_plant += mask_no.sum().item()
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    result = {
        "loss": avg_loss,
        "accuracy": avg_acc,
    }
    
    if task == "winner":
        result["accuracy_has_plant"] = correct_has_plant / total_has_plant if total_has_plant > 0 else 0
        result["accuracy_no_plant"] = correct_no_plant / total_no_plant if total_no_plant > 0 else 0
    
    return result


def train_epoch(model, loader, criterion, optimizer, scheduler, device, task):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        if task == "winner":
            inputs, labels, has_plant = batch
            inputs, labels = inputs.to(device), labels.to(device)
        else:
            inputs, labels, _ = batch
            inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).float().sum().item()
        total_samples += inputs.size(0)
    
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / total_samples, total_correct / total_samples


def main():
    args = get_args()
    
    # setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    output_dir = Path(args.output_dir) / f"{args.task}_{args.mode}_{args.model}_{args.img_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    writer = SummaryWriter(output_dir / "logs")
    
    # data
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(args)
    
    # model
    print("Creating model...")
    model = create_model(args).to(device)
    
    # training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"Training {args.task} model on {device}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    for epoch in range(args.epochs):
        start = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, args.task
        )
        
        val_result = evaluate(model, val_loader, criterion, device, args.task)
        
        epoch_time = time.time() - start
        
        # log
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_result["loss"], epoch)
        writer.add_scalar("val/accuracy", val_result["accuracy"], epoch)
        
        if args.task == "winner":
            writer.add_scalar("val/accuracy_has_plant", val_result["accuracy_has_plant"], epoch)
            writer.add_scalar("val/accuracy_no_plant", val_result["accuracy_no_plant"], epoch)
        
        # save best
        if val_result["accuracy"] > best_val_acc:
            best_val_acc = val_result["accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        
        # print
        log_str = f"Epoch {epoch+1}/{args.epochs} | {epoch_time:.1f}s | "
        log_str += f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
        log_str += f"Val: loss={val_result['loss']:.4f} acc={val_result['accuracy']:.4f}"
        
        if args.task == "winner":
            log_str += f" | has_plant={val_result['accuracy_has_plant']:.4f} no_plant={val_result['accuracy_no_plant']:.4f}"
        
        print(log_str)
    
    print(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    writer.close()


if __name__ == "__main__":
    main()
