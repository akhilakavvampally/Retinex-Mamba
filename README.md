# Retinex-Mamba
To Enhance the images captured under different Low light conditions by combining the Retienx Theory and DeepLearning to create a Retinex Mamba
 # 1.Clone the repository
git clone https://github.com/yourusername/RetinexMamba.git
cd RetinexMamba
# 2.Install dependencies
pip install -r requirements.txt
#If requirements.txt is missing, ensure the following packages are installed:
torch torchvision opencv-python numpy pillow tqdm scikit-image tensorboard

# 3.Train the Model
python train.py \
  --input_dir /path/to/Low \
  --target_dir /path/to/Normal \
  --epochs 20 \
  --batch_size 4 \
  --lr 1e-4 \
  --save_path ./retinexmamba_trained.pth \
  --log_dir ./runs \
  --output_vis_dir ./validation_outputs
# 4.Tensor board
tensorboard --logdir=./runs
# 5.Model Saving
#After training completes, the model is saved to the path specified via --save_path.

torch.save(model.state_dict(), 'retinexmamba_trained.pth')
