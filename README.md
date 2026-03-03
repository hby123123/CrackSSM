# CrackSSM

SSM-based crack segmentation. Test on TUT, DeepCrack, Crack500. Python 3.10, PyTorch 2.4.1, CUDA 12.1.

---

## 1.Create environment

## Important: CUDA & NVCC Version Check
This project requires the **selective_scan** CUDA extension. To ensure successful compilation, your system's NVCC version **must** match your PyTorch CUDA version (**12.1**).
Before running the setup script, please verify:
```bash
nvcc --version  # Ensure it shows 12.1
python -c "import torch; print(torch.version.cuda)"  # Ensure it matches 12.1\
```
```bash
git clone https://github.com/hby123123/CrackSSM
cd CrackSSM
conda create -n crackssm python=3.10 -y && conda activate crackssm
chmod +x scripts/setup_env.sh && ./scripts/setup_env.sh
```
You can run it from the repo root (as above) or with an absolute path: `bash /path/to/CrackSSM/scripts/setup_env.sh`. It installs PyTorch 2.4.1+cu121, dependencies from `requirements.txt`, and the selective_scan CUDA extension.

If the `crackssm` conda env already exists, activate it and run only the script:

```bash
cd CrackSSM
conda activate crackssm
./scripts/setup_env.sh
```
---

## 2. Dataset layout

Put data under `data/` with layout below (TUT, DeepCrack, Crack500):

```
data/<dataset_name>/
├── train_img/    # .jpg
├── train_lab/    # .png or .bmp, binary (>128 foreground)
├── val_img/
├── val_lab/
├── test_img/
└── test_lab/
```

---

## 3. Train

Generate soft directional pseudo labels from `train_lab`:

```bash
python train/train.py --dataset TUT --data_root ./data --generate_labels
```
Then train (with or without resume):
```bash
python train/train.py --dataset TUT --data_root ./data --output_dir checkpoints/TUT
python train/train.py --dataset TUT --resume checkpoints/TUT/best_model.pth --pseudo_weight 0.5
```

## 4. Test 
Test loads a local checkpoint and writes prediction images.
```bash
./scripts/run_test.sh TUT
./scripts/run_test.sh DeepCrack
./scripts/run_test.sh Crack500
```
Or with Python (optional `--ckpt` for custom checkpoint path):
```bash
python test/run_test.py --dataset TUT
python test/run_test.py --dataset DeepCrack --ckpt checkpoints/DeepCrack/best_model.pth
```

## 5. Evaluate 
Metrics (mIoU, ODS, OIS, F1, Precision, Recall) are computed  by the evaluation script, using saved predictions and ground truth:
```bash
python eval/evaluation.py --gt_dir data/TUT/test_lab --pred_dir test/test_results/TUT
```
Run test first to generate `test/test_results/<dataset>/`, then point `--pred_dir` to that folder and `--gt_dir` to the corresponding test labels.
