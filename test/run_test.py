import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ALLOWED_DATASETS = ("TUT", "DeepCrack", "Crack500")


def load_checkpoint(model, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if not isinstance(state, dict):
        raise ValueError("checkpoint format not recognized")
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def main():
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from config import PROJECT_ROOT, get_data_root, DEFAULT_INPUT_SIZE
    from train.dataset import CrackDataset, collate_train
    from models.model import Model

    parser = argparse.ArgumentParser(description="CrackSSM test (TUT, DeepCrack, Crack500), inference only, use eval for metrics.")
    parser.add_argument("--dataset", type=str, nargs="+", default=["TUT"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    for d in args.dataset:
        if d not in ALLOWED_DATASETS:
            parser.error(f"--dataset must be one or more of {ALLOWED_DATASETS}, got: {d}")

    data_root = get_data_root(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.ckpt is None:
        args.ckpt = os.path.join(PROJECT_ROOT, "checkpoints", args.dataset[0], "best_model.pth")
    if args.save_dir is None:
        args.save_dir = os.path.join(PROJECT_ROOT, "test", "test_results")

    model = Model().to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    for dataset_name in args.dataset:
        dataset_path = os.path.join(data_root, dataset_name)
        if not os.path.isdir(dataset_path):
            if os.path.isdir(data_root) and os.path.isdir(os.path.join(data_root, "test_img")):
                dataset_path = data_root
                dataset_name = os.path.basename(data_root.rstrip("/")) or "default"
            else:
                print(f"skip: dataset dir not found: {dataset_path}")
                continue
        test_img_dir = os.path.join(dataset_path, "test_img")
        if not os.path.isdir(test_img_dir):
            print(f"skip: no test_img in {dataset_path}")
            continue
        save_subdir = os.path.join(args.save_dir, dataset_name)
        if not args.no_save:
            os.makedirs(save_subdir, exist_ok=True)

        test_set = CrackDataset(dataset_path, mode="test", augment=False, input_size=DEFAULT_INPUT_SIZE)
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_train
        )

        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Test {dataset_name}", leave=True)
            for images, _, _, filenames in pbar:
                images = images.to(device)
                pred_out, _ = model(images)
                pred_probs = torch.sigmoid(pred_out)
                pred_np = pred_probs.cpu().numpy()
                if not args.no_save:
                    for i in range(pred_np.shape[0]):
                        pred_uint8 = (pred_np[i].squeeze() * 255).astype(np.uint8)
                        base_name, _ = os.path.splitext(filenames[i])
                        save_filepath = os.path.join(save_subdir, f"{base_name}.png")
                        Image.fromarray(pred_uint8).save(save_filepath)

        print(f"[{dataset_name}] samples: {len(test_set)}")
        if not args.no_save:
            print(f"  predictions saved: {os.path.abspath(save_subdir)}")


if __name__ == "__main__":
    main()
