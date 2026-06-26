from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


CLASSES = ["rainy_muddy", "snowy_offroad", "dusty_sandy"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalize_class_name(name: str) -> str | None:
    low = name.lower()
    for cls in CLASSES:
        if low == cls or low.startswith(cls):
            return cls
    return None


def collect_split(root: Path, split: str) -> list[tuple[Path, int, str]]:
    rows: list[tuple[Path, int, str]] = []
    for idx, cls in enumerate(CLASSES):
        d = root / split / cls
        if not d.exists():
            raise FileNotFoundError(f"Missing real split folder: {d}")
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                rows.append((p, idx, "real"))
    return rows


def collect_generated(root: Path | None) -> list[tuple[Path, int, str]]:
    if root is None:
        return []
    rows: list[tuple[Path, int, str]] = []
    if not root.exists():
        raise FileNotFoundError(f"Generated root not found: {root}")
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        cls = normalize_class_name(child.name)
        if cls is None:
            continue
        idx = CLASSES.index(cls)
        for p in sorted(child.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                rows.append((p, idx, "generated"))
    return rows


class ImageListDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, int, str]], transform=None) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, label, source = self.rows[idx]
        with Image.open(path) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(path), source


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def build_model(num_classes: int, pretrained: bool, freeze_backbone: bool = False) -> nn.Module:
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except Exception as exc:
            print(f"[WARN] Could not load ImageNet weights, using random init: {exc}")
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true, y_pred = [], []
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=list(range(len(CLASSES))))
    per = f1_score(y_true, y_pred, average=None, labels=list(range(len(CLASSES))))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    return {
        "acc": float(acc),
        "macro_f1": float(macro),
        "per_class_f1": {cls: float(per[i]) for i, cls in enumerate(CLASSES)},
        "worst_class_f1": float(per.min()),
        "confusion_matrix": cm.tolist(),
    }


def train_one_run(args, run_dir: Path, seed: int, generated_rows: list[tuple[Path, int, str]]) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_tf, eval_tf = build_transforms()
    real_root = Path(args.real_root)

    train_rows = collect_split(real_root, "train") + generated_rows
    val_rows = collect_split(real_root, "val")
    test_rows = collect_split(real_root, "test")

    train_ds = ImageListDataset(train_rows, train_tf)
    val_ds = ImageListDataset(val_rows, eval_tf)
    test_ds = ImageListDataset(test_rows, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(len(CLASSES), pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_macro = -1.0
    best_path = run_dir / f"best_seed{seed}.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels, _, _ in tqdm(train_loader, desc=f"seed {seed} epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": total_loss / max(len(train_ds), 1), **val_metrics})
        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    result = {
        "seed": seed,
        "device": str(device),
        "train_count": len(train_rows),
        "generated_train_count": len(generated_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "history": history,
        "test": test_metrics,
        "best_checkpoint": str(best_path),
    }
    (run_dir / f"metrics_seed{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def save_confusion_matrix(cm: list[list[int]], out_path: Path) -> None:
    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize(results: list[dict], out_dir: Path, experiment_name: str) -> dict:
    rows = []
    cms = []
    for r in results:
        test = r["test"]
        row = {
            "experiment": experiment_name,
            "seed": r["seed"],
            "acc": test["acc"],
            "macro_f1": test["macro_f1"],
            "worst_class_f1": test["worst_class_f1"],
        }
        for cls, val in test["per_class_f1"].items():
            row[f"{cls}_f1"] = val
        rows.append(row)
        cms.append(np.array(test["confusion_matrix"]))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    mean_row = df.drop(columns=["seed"]).groupby("experiment").agg(["mean", "std"])
    mean_row.to_csv(out_dir / "summary_mean_std.csv")
    cm_mean = np.rint(np.mean(cms, axis=0)).astype(int).tolist()
    save_confusion_matrix(cm_mean, out_dir / "confusion_matrix_mean.png")
    return {
        "experiment": experiment_name,
        "per_seed": rows,
        "mean": df.drop(columns=["experiment", "seed"]).mean(numeric_only=True).to_dict(),
        "std": df.drop(columns=["experiment", "seed"]).std(numeric_only=True).fillna(0).to_dict(),
        "confusion_matrix_mean_rounded": cm_mean,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", default="data/downstream/real_curated")
    parser.add_argument("--generated-root", default=None, help="Optional generated image root with class subfolders. Added to train only.")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--output-root", default="outputs/downstream_resnet18")
    parser.add_argument("--classes", nargs="+", default=CLASSES)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    global CLASSES
    args = parse_args()
    CLASSES = args.classes
    generated_root = Path(args.generated_root) if args.generated_root else None
    generated_rows = collect_generated(generated_root)
    default_name = "real_plus_generated" if generated_rows else "real_only"
    experiment_name = args.experiment_name or default_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"{stamp}_{experiment_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["generated_count"] = len(generated_rows)
    config["classes"] = CLASSES
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = []
    for seed in args.seeds:
        results.append(train_one_run(args, out_dir, seed, generated_rows))
    summary = summarize(results, out_dir, experiment_name)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
