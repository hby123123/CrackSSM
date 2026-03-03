import os
import sys
import random
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

DEFAULT_EPOCHS = 400
PSEUDO_DIR_NAME = "train_pseudo_labels_soft"
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 8e-4
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42
AUX_LOSS_ALPHA = 0.5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.max_iters = max_iters
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / self.max_iters) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", zero_logits_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.zero_logits_weight = zero_logits_weight

    def forward(self, pred_logits, target_probs):
        log_probs = F.log_softmax(pred_logits, dim=1)
        mask_valid = (target_probs.sum(dim=1) > 0).float()
        mask_masked = (target_probs.sum(dim=1) == 0).float()
        soft_ce_per_sample = -torch.sum(target_probs * log_probs, dim=1)
        main_loss = soft_ce_per_sample * mask_valid
        zero_logits_force_loss = torch.tensor(0.0, device=pred_logits.device)
        if self.zero_logits_weight > 0 and torch.any(mask_masked):
            zero_logits_force_loss = self.zero_logits_weight * (pred_logits ** 2).sum(dim=1) * mask_masked
        total_per_sample = main_loss + zero_logits_force_loss
        if self.reduction == "mean":
            return total_per_sample.mean()
        if self.reduction == "sum":
            return total_per_sample.sum()
        return total_per_sample


def create_soft_directional_pseudo_labels(mask_path, save_path, temperature=0.1):
    try:
        import cv2
        from scipy.special import softmax
    except ImportError:
        raise ImportError("create_soft_directional_pseudo_labels requires opencv-python and scipy")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return
    mask = (mask > 128).astype(np.float32)
    k_horz = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
    k_vert = np.transpose(k_horz)
    k_diag45 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
    k_diag135 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)
    responses = [cv2.filter2D(mask, -1, k) for k in [k_horz, k_vert, k_diag45, k_diag135]]
    responses_tensor = np.stack(responses, axis=-1)
    responses_tensor[responses_tensor < 0] = 0
    mask_pixels = mask > 0
    soft_pseudo_label = np.zeros_like(responses_tensor, dtype=np.float32)
    if np.any(mask_pixels):
        pixel_responses = responses_tensor[mask_pixels]
        soft_pixel_labels = softmax(pixel_responses / temperature, axis=-1)
        soft_pseudo_label[mask_pixels] = soft_pixel_labels
    np.save(save_path, soft_pseudo_label)


def run_pseudo_label_generation(dataset_path, pseudo_dir, label_ext=(".png", ".bmp")):
    lab_dir = os.path.join(dataset_path, "train_lab")
    if not os.path.isdir(lab_dir):
        raise FileNotFoundError(f"train_lab not found: {lab_dir}")
    os.makedirs(pseudo_dir, exist_ok=True)
    label_files = []
    for ext in label_ext:
        label_files.extend([f for f in os.listdir(lab_dir) if f.endswith(ext)])
    label_files = sorted(set(label_files))
    for filename in tqdm(label_files, desc="Generating soft pseudo labels"):
        mask_path = os.path.join(lab_dir, filename)
        save_path = os.path.join(pseudo_dir, os.path.splitext(filename)[0] + ".npy")
        create_soft_directional_pseudo_labels(mask_path, save_path, temperature=0.1)
    print(f"Saved soft pseudo labels to {pseudo_dir} ({len(label_files)} files)")


def main():
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from config import get_data_root, get_ckpt_dir, DEFAULT_INPUT_SIZE
    from train.dataset import CrackDataset, collate_train
    from models.model import Model

    parser = argparse.ArgumentParser(description="CrackSSM train (with optional soft directional pseudo labels)")
    parser.add_argument("--dataset", type=str, default="TUT")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pseudo_dir", type=str, default=None)
    parser.add_argument("--pseudo_weight", type=float, default=AUX_LOSS_ALPHA)
    parser.add_argument("--generate_labels", action="store_true", help="Generate soft directional pseudo labels from train_lab and exit")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = get_data_root(args.data_root)
    dataset_path = os.path.join(data_root, args.dataset)
    if not os.path.isdir(dataset_path) or not os.path.isdir(os.path.join(dataset_path, "train_img")):
        if os.path.isdir(os.path.join(data_root, "train_img")):
            dataset_path = data_root
        else:
            raise FileNotFoundError(f"Dataset dir not found: {dataset_path} (need train_img/train_lab)")
    output_dir = args.output_dir or get_ckpt_dir(args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    epoch_ckpt_path = os.path.join(output_dir, "epoch_ckpts")
    os.makedirs(epoch_ckpt_path, exist_ok=True)

    pseudo_dir = args.pseudo_dir or os.path.join(dataset_path, PSEUDO_DIR_NAME)
    if not os.path.isabs(pseudo_dir) and args.pseudo_dir is None:
        pseudo_dir = os.path.join(dataset_path, PSEUDO_DIR_NAME)
    if args.pseudo_dir and os.path.isabs(args.pseudo_dir):
        pseudo_dir = args.pseudo_dir

    if args.generate_labels:
        run_pseudo_label_generation(dataset_path, pseudo_dir)
        return

    if not os.path.isdir(pseudo_dir) or not os.listdir(pseudo_dir):
        print(f"Pseudo dir empty or missing: {pseudo_dir}. Run with --generate_labels first; aux loss will use zeros.")
        pseudo_dir = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Data: {dataset_path}, Output: {output_dir}")

    train_set = CrackDataset(
        dataset_path, mode="train", augment=True, input_size=DEFAULT_INPUT_SIZE, pseudo_dir=pseudo_dir
    )
    val_dir = os.path.join(dataset_path, "val_img")
    test_dir = os.path.join(dataset_path, "test_img")
    if os.path.isdir(val_dir):
        val_set = CrackDataset(dataset_path, mode="val", augment=False, input_size=DEFAULT_INPUT_SIZE, pseudo_dir=pseudo_dir or "dummy_path")
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_train)
    else:
        val_set = None
        val_loader = None
    if os.path.isdir(test_dir):
        test_set = CrackDataset(dataset_path, mode="test", augment=False, input_size=DEFAULT_INPUT_SIZE, pseudo_dir=pseudo_dir or "dummy_path")
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_train)
    else:
        test_set = None
        test_loader = None
    if val_loader is None and test_loader is None:
        fallback = CrackDataset(dataset_path, mode="train", augment=False, input_size=DEFAULT_INPUT_SIZE, pseudo_dir=pseudo_dir or "dummy_path")
        val_loader = DataLoader(fallback, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_train)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate_train
    )

    _val_len = len(val_loader.dataset) if val_loader else 0
    _test_len = len(test_loader.dataset) if test_loader else 0
    print(f"Train: {len(train_set)}, Val: {_val_len}, Test: {_test_len}")

    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    max_iters = args.epochs * len(train_loader)
    scheduler = PolyLRScheduler(optimizer, max_iters=max_iters)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    criterion_aux = SoftCrossEntropyLoss().to(device)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            start_epoch = ckpt.get("epoch", 0) + 1
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
            start_epoch = ckpt.get("epoch", 0) + 1
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}")

    log_path = os.path.join(output_dir, "training_log.csv")
    _log_header = ["epoch", "train_loss_total", "train_loss_main", "train_loss_aux", "val_loss_total", "val_loss_main", "val_loss_aux", "learning_rate"]
    if test_loader is not None:
        _log_header = _log_header + ["test_loss_total", "test_loss_main", "test_loss_aux"]
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_log_header)

    best_loss_main = float("inf")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = {"total": 0.0, "main": 0.0, "aux": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for images, labels, pseudo_labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            pseudo_labels = pseudo_labels.to(device)
            optimizer.zero_grad()
            pred_out, orientation_map = model(images)
            loss_main = 3.0 * criterion_bce(pred_out, labels) + criterion_dice(pred_out, labels)
            loss_aux = criterion_aux(orientation_map, pseudo_labels)
            loss = loss_main + args.pseudo_weight * loss_aux
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses["total"] += loss.item()
            losses["main"] += loss_main.item()
            losses["aux"] += loss_aux.item()
            pbar.set_postfix(loss=f"{losses['total'] / (pbar.n + 1):.4f}", aux=f"{losses['aux'] / (pbar.n + 1):.4f}")

        train_loss = {k: v / len(train_loader) for k, v in losses.items()}

        model.eval()
        val_losses = {"total": 0.0, "main": 0.0, "aux": 0.0}
        with torch.no_grad():
            for images, labels, pseudo_labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                pseudo_labels = pseudo_labels.to(device)
                pred_out, orientation_map = model(images)
                loss_main = 3.0 * criterion_bce(pred_out, labels) + criterion_dice(pred_out, labels)
                loss_aux = criterion_aux(orientation_map, pseudo_labels) if pseudo_labels.abs().sum() > 0 else torch.tensor(0.0, device=device)
                loss = loss_main + args.pseudo_weight * loss_aux
                val_losses["total"] += loss.item()
                val_losses["main"] += loss_main.item()
                val_losses["aux"] += (loss_aux.mean().item() if loss_aux.dim() > 0 else loss_aux.item())
        val_loss = {k: v / len(val_loader) for k, v in val_losses.items()}

        test_loss = None
        if test_loader is not None:
            test_losses = {"total": 0.0, "main": 0.0, "aux": 0.0}
            with torch.no_grad():
                for images, labels, pseudo_labels, _ in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    pseudo_labels = pseudo_labels.to(device)
                    pred_out, orientation_map = model(images)
                    loss_main = 3.0 * criterion_bce(pred_out, labels) + criterion_dice(pred_out, labels)
                    loss_aux = criterion_aux(orientation_map, pseudo_labels) if pseudo_labels.abs().sum() > 0 else torch.tensor(0.0, device=device)
                    loss = loss_main + args.pseudo_weight * loss_aux
                    test_losses["total"] += loss.item()
                    test_losses["main"] += loss_main.item()
                    test_losses["aux"] += (loss_aux.mean().item() if loss_aux.dim() > 0 else loss_aux.item())
            test_loss = {k: v / len(test_loader) for k, v in test_losses.items()}

        lr_now = optimizer.param_groups[0]["lr"]
        _log_row = [epoch + 1, train_loss["total"], train_loss["main"], train_loss["aux"], val_loss["total"], val_loss["main"], val_loss["aux"], lr_now]
        if test_loss is not None:
            _log_row += [test_loss["total"], test_loss["main"], test_loss["aux"]]
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_log_row)

        loss_for_best = (test_loss["main"] if test_loss is not None else val_loss["main"])
        print(
            f"Epoch [{epoch+1}/{args.epochs}] LR: {lr_now:.6f} | "
            f"Train: {train_loss['total']:.4f} (main: {train_loss['main']:.4f}, aux: {train_loss['aux']:.4f}) | "
            f"Val: {val_loss['total']:.4f} (main: {val_loss['main']:.4f}, aux: {val_loss['aux']:.4f})"
            + (f" | Test: {test_loss['total']:.4f} (main: {test_loss['main']:.4f})" if test_loss is not None else "")
        )

        torch.save(model.state_dict(), os.path.join(epoch_ckpt_path, f"model_epoch_{epoch+1:03d}.pth"))
        if loss_for_best < best_loss_main:
            best_loss_main = loss_for_best
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            _which = "test" if test_loss is not None else "val"
            print(f"  -> Saved best model ({_which} main loss: {best_loss_main:.4f})")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, os.path.join(output_dir, "last_model.pth"))

    _best_criterion = "test" if test_loader is not None else "val"
    print(f"Training finished. Best model (by {_best_criterion} main loss) saved to", os.path.join(output_dir, "best_model.pth"))


if __name__ == "__main__":
    main()
