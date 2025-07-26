# RetinexMamba Low-Light Image Enhancement

This repository contains a PyTorch-based implementation of the **RetinexMamba** architecture for low-light image enhancement using illumination estimation and damage restoration techniques.

## 📌 Features

- **Illumination Estimation Module** – Enhances image illumination.
- **Damage Restorer Module** – Recovers details from poorly lit regions.
- Supports **SSIM** and **MSE** based training.
- Integrated **TensorBoard** logging.
- Includes **validation PSNR/SSIM evaluation** and side-by-side output visualization.

---

## 🗂️ Project Structure

```
.
├── model.py            # RetinexMamba model definition
├── train.py            # Training and evaluation logic
├── dataset/            # Input low-light and target normal-light images
├── runs/               # TensorBoard logs
├── validation_outputs/ # Evaluation results
└── retinexmamba_trained.pth  # Trained model weights (output)
```

---

## 📥 Dataset

This model is compatible with datasets like [LOLv1/LOLv2](https://daooshee.github.io/BMVC2018website/) having paired low-light and normal-light images.

Update the dataset paths in `main()`:

```python
parser.add_argument('--input_dir', default='/path/to/low_light_images')
parser.add_argument('--target_dir', default='/path/to/normal_light_images')
```

---

## 🚀 How to Run in Google Colab

```python
!pip install torch torchvision tensorboard scikit-image pillow opencv-python
!python train.py
```

In `train.py`, `argparse` is initialized with `args = parser.parse_args([])` to support Colab environment.

---

## 🧠 Training

```bash
python train.py --epochs 20 --batch_size 4 --lr 1e-4   --input_dir /path/to/low --target_dir /path/to/normal   --save_path ./retinexmamba_trained.pth
```

TensorBoard logs will be saved to `/content/runs` or the specified `--log_dir`.

---

## 🖼️ Output Visualization

Validation samples (input/enhanced/ground-truth triplets) are saved in `/content/validation_outputs` by default. You can change this using:

```bash
--output_vis_dir /your/custom/output/dir
```

---

## 📊 Evaluation Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- These are logged to TensorBoard and printed every epoch.

---

## 🧾 License

This project is under the MIT License.

---

## ✍️ Author

Developed by [Your Name]. Contributions and issues are welcome!

