# URDF-Anything+: Autoregressive Articulated 3D Models Generation for Physical Simulation
<p align="center">
  <a href="https://urdf-anything-plus.github.io/"><img alt="Website" src="https://img.shields.io/badge/Website-Project%20Page-1f2937?style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="https://arxiv.org/pdf/2603.14010"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.14010-b31b1b?style=flat-square&logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/URDF-Anything-plus/Dataset"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-Hugging%20Face-ffb000?style=flat-square&logo=huggingface&logoColor=000"></a>
  <a href="https://huggingface.co/URDF-Anything-plus/URDF-Anything-Plus-Model"><img alt="Model" src="https://img.shields.io/badge/Model-Hugging%20Face-3b82f6?style=flat-square&logo=huggingface&logoColor=white"></a>
</p>

<div align="center">
  <div>
    <span class="author-block">
      <a href="https://zhuangzhewu.github.io/">Zhuangzhe Wu</a><sup>1</sup>,</span>
    <span class="author-block">
      <a href="https://github.com/vinkda">Yue Xin</a><sup>1</sup>,</span>
    <span class="author-block">
      <a href="https://jackhck.github.io/">Chengkai Hou</a><sup>1</sup>,</span>
    <span class="author-block">
      <a href="https://silent-chen.github.io/">Minghao Chen</a><sup>2</sup>,</span>
    <span class="author-block">
      <a href="https://scholar.google.com/citations?user=cpPgzGkAAAAJ">Yaoxu Lyu</a><sup>1</sup>,</span>
    <span class="author-block">
      <a href="https://jieyuz2.github.io/">Jieyu Zhang</a><sup>3</sup>,</span>
    <span class="author-block">
      <a href="https://scholar.google.com/citations?hl=en&user=voqw10cAAAAJ&view_op=list_works&sortby=pubdate">Shanghang Zhang</a><sup>1</sup></span>
  </div>
</div>

## Requirements
### Python environment
1. **Clone the repository**
   ```bash
   git clone https://github.com/URDF-Anything-plus/URDF-Anything-plus.git
   cd URDF-Anything-plus
   ```
2. **Create a conda environment**
   ```bash
   conda create -n urdf-anything python=3.10 -y
   conda activate urdf-anything
   ```

3. **Install PyTorch**
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 
   ```

4. **Install dependencies**：
   ```bash
   pip install -r  -u requirements.txt -i https://pypi.org/simple/
   ```

5. **Install torch-cluster**。Must be installed after PyTorch:
   ```bash
   pip install torch-cluster --no-build-isolation
   ```
   
6. **Install diso**（TripoSG mesh extraction will use it）. Must be installed after PyTorch:
   ```bash
   pip install diso --no-build-isolation
   ```

### Pretrained Models (TripoSG & DINOv3)

**Hugging Face authentication**（recommended to configure before downloading models）：


**Setup:** Clone [TripoSG](https://github.com/VAST-AI-Research/TripoSG) (used for 3D geometry)，and download the weights in `TripoSG/pretrained_weights/`：

```bash
# 1) Clone TripoSG code
git clone https://github.com/VAST-AI-Research/TripoSG.git

# 2) Download TripoSG main model (contains transformer / vae / model_index.json etc.)
huggingface-cli download VAST-AI/TripoSG --local-dir TripoSG/pretrained_weights/TripoSG

# 3) Download RMBG-1.4 background removal model
huggingface-cli download briaai/RMBG-1.4 --local-dir TripoSG/pretrained_weights/RMBG-1.4

# 4) Download DINOv3 image encoder (used for cache building and inference)
huggingface-cli download facebook/dinov3-vith16plus-pretrain-lvd1689m --local-dir DINOv3
```

If `huggingface-cli` is not installed, you can also download the models using Python:

```bash
python -c "
from huggingface_hub import snapshot_download

# TripoSG 
snapshot_download(repo_id='VAST-AI/TripoSG', local_dir='TripoSG/pretrained_weights/TripoSG')

# RMBG-1.4
snapshot_download(repo_id='briaai/RMBG-1.4', local_dir='TripoSG/pretrained_weights/RMBG-1.4')

# DINOv3
snapshot_download(repo_id='facebook/dinov3-vith16plus-pretrain-lvd1689m', local_dir='DINOv3')
"
```

Important: There is a little problem with TripoSG/triposg/models/autoencoders/autoencoder_kl_triposg.py, you need to uncomment the line `from torch_cluster import fps`.

## Training
### Data Preparation
Download the dataset from [Hugging Face](https://huggingface.co/datasets/URDF-Anything-plus/Dataset) and unzip it to `data_normalized/`.

The structure of the dataset is as follows:
```
URDF-Anything-plus:
├── data_normalized/
│   ├── Laptop_urdf/
│   │   ├── images/
│   │   ├── xxx.obj
│   │   ├── test.urdf/
│   │   ├── info.json/
│   ├── Refrigerator_urdf/
│   │   ├── images/
│   │   ├── xxx.obj
│   │   ├── test.urdf/
│   │   ├── info.json/
│   ├── ...
```

Then run the following command to build the cache:
```bash
python scripts/build_cache.py
```

### Training
```bash
bash scripts/run_multi_node_training.sh [node_rank] [master_addr] [nproc_per_node] [training parameters...]
```
For example, to train on 1 machine with 8 GPUs, you can run:
```bash
bash scripts/run_multi_node_training.sh 0 localhost 8
```
You can adjust the training parameters in `scripts/run_multi_node_training.sh`.

In pretraining stage, we use the following hyperparameters:
```yaml
--init_mode train_from_scratch
```
In finetuning stage, we use the following hyperparameters:
```yaml
--init_mode resume_from_ckpt
--checkpoint_path CHECKPOINT_PATH FROM PRETRAINING STAGE
--train_urdf_params True
--train_eot True
```

## Inference

You can try our inference script:
```bash
bash scripts/inference.sh
```

If you are in 'in_the_wild' mode, you should make sure the object is oriented towards the positive z direction. See the examples below, the z-axis is the blue line. 
<p align="center">
  <img src="assets/laptop_example.png" alt="Laptop Example" width="33%" style="display:inline-block" />
  <img src="assets/display_example.png" alt="Display Example" width="30%" style="display:inline-block;" />
  <img src="assets/faucet_example.png" alt="Faucet Example" width="30%" style="display:inline-block" />
</p>

You can rotate the mesh in the terminal to check the orientation. The rotation commands are as follows:
```
Optional: rotate the mesh in the terminal. You need to make sure the object is oriented towards the positive z direction. You can rotate multiple times, press Enter to end.
  0: no rotation (pressing Enter is equivalent to ending)
  1-6: rotate around X/Y/Z axis ±90°
angle_map_print: {'1': 'rotate around X axis 90°', '2': 'rotate around X axis -90°', '3': 'rotate around Y axis 90°', '4': 'rotate around Y axis -90°', '5': 'rotate around Z axis 90°', '6': 'rotate around Z axis -90°'}
Enter the rotation number and press Enter (pressing Enter is equivalent to ending): [0-6]:
```
You need to make sure the object is oriented towards the positive z direction.
