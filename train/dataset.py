import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

try:
    from config import DEFAULT_INPUT_SIZE
except ImportError:
    DEFAULT_INPUT_SIZE = [448, 448]
class CrackDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode="train",
        augment=False,
        input_size=None,
        label_ext=None,
        pseudo_dir=None,
    ):
        self.root_dir = os.path.expanduser(root_dir)
        self.mode = mode
        self.augment = augment and (mode == "train")
        self.input_size = list(input_size or DEFAULT_INPUT_SIZE)
        self.label_ext = label_ext or (".png", ".bmp")
        self.pseudo_dir = os.path.join(self.root_dir, pseudo_dir) if pseudo_dir and not os.path.isabs(pseudo_dir) else pseudo_dir

        self.img_dir = os.path.join(self.root_dir, f"{mode}_img")
        self.lab_dir = os.path.join(self.root_dir, f"{mode}_lab")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")

        self.image_names = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        if not self.image_names:
            raise FileNotFoundError(f"No image files in {self.img_dir}")

        self.photometric_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        ])
        self.to_tensor_and_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_names)

    def _load_label(self, base_name):
        for ext in self.label_ext:
            p = os.path.join(self.lab_dir, base_name + ext)
            if os.path.isfile(p):
                return Image.open(p).convert("L")
        raise FileNotFoundError(f"Label not found: {base_name} ({self.lab_dir})")

    def _load_pseudo(self, base_name, h, w):
        out = np.zeros((h, w, 4), dtype=np.float32)
        if self.pseudo_dir and os.path.isdir(self.pseudo_dir):
            p = os.path.join(self.pseudo_dir, base_name + ".npy")
            if os.path.isfile(p):
                raw = np.load(p).astype(np.float32)
                if raw.ndim == 3 and raw.shape[2] == 4:
                    if raw.shape[0] == h and raw.shape[1] == w:
                        out = raw
                    else:
                        out = np.zeros((h, w, 4), dtype=np.float32)
                        for c in range(4):
                            out[:, :, c] = np.array(TF.resize(Image.fromarray((raw[:, :, c] * 255).astype(np.uint8)), (h, w), TF.InterpolationMode.BILINEAR)) / 255.0
                elif raw.ndim == 2:
                    out = np.stack([raw.astype(np.float32) / 255.0] * 4, axis=-1)
        return out

    def __getitem__(self, idx):
        image_filename = self.image_names[idx]
        base_name = os.path.splitext(image_filename)[0]
        img_path = os.path.join(self.img_dir, image_filename)
        image = Image.open(img_path).convert("RGB")
        label = self._load_label(base_name)
        w, h = label.size
        pseudo_label = self._load_pseudo(base_name, h, w)

        if self.augment:
            i, j, crop_h, crop_w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.5, 1.0), ratio=(0.75, 1.33)
            )
            image = TF.resized_crop(image, i, j, crop_h, crop_w, self.input_size, TF.InterpolationMode.BILINEAR)
            label = TF.resized_crop(label, i, j, crop_h, crop_w, self.input_size, TF.InterpolationMode.BICUBIC)
            pseudo_pil = [Image.fromarray((pseudo_label[:, :, c] * 255).astype(np.uint8)) for c in range(4)]
            pseudo_pil = [TF.resized_crop(p, i, j, crop_h, crop_w, self.input_size, TF.InterpolationMode.BILINEAR) for p in pseudo_pil]
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                pseudo_pil = [TF.hflip(p) for p in pseudo_pil]
            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
                pseudo_pil = [TF.vflip(p) for p in pseudo_pil]
            if random.random() > 0.5:
                angle = transforms.RandomRotation.get_params([-30, 30])
                image = TF.rotate(image, angle, TF.InterpolationMode.BILINEAR)
                label = TF.rotate(label, angle, TF.InterpolationMode.BICUBIC)
                pseudo_pil = [TF.rotate(p, angle, TF.InterpolationMode.BILINEAR) for p in pseudo_pil]
            image = self.photometric_transform(image)
            pseudo_np = np.stack([np.array(p) / 255.0 for p in pseudo_pil], axis=2).astype(np.float32)
        else:
            image = TF.resize(image, self.input_size, TF.InterpolationMode.BILINEAR)
            label = TF.resize(label, self.input_size, TF.InterpolationMode.BICUBIC)
            pseudo_pil = [Image.fromarray((pseudo_label[:, :, c] * 255).astype(np.uint8)) for c in range(4)]
            pseudo_pil = [TF.resize(p, self.input_size, TF.InterpolationMode.BILINEAR) for p in pseudo_pil]
            pseudo_np = np.stack([np.array(p) / 255.0 for p in pseudo_pil], axis=2).astype(np.float32)

        image_tensor = self.to_tensor_and_norm(image)
        label_tensor = (self.to_tensor(label) > 0.5).to(torch.float32)
        pseudo_label_tensor = torch.from_numpy(pseudo_np).permute(2, 0, 1).float()
        return image_tensor, label_tensor, pseudo_label_tensor, base_name


def collate_train(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pseudos = torch.stack([b[2] for b in batch])
    names = [b[3] for b in batch]
    return images, labels, pseudos, names


TUTDataset = CrackDataset
