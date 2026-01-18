# SMoENet
# Sparse Mixture-of-Experts Network with Modal Interaction for Multimodal Emotion Recognition

This is an official implementation of 'Sparse Mixture-of-Experts Network with Modal Interaction for Multimodal Emotion Recognition' :fire:.

<div  align="center"> 
  <img src="https://github.com/Xudmm1239439/SMoENet/blob/main/SMOE.png" width=100% />
</div>



## 🚀 Installation

```bash
certifi               2025.7.14
charset-normalizer    3.4.2
clip                  1.0
contourpy             1.1.1
cycler                0.12.1
fonttools             4.57.0
ftfy                  6.2.3
googledrivedownloader 1.1.0
idna                  3.10
importlib_resources   6.4.5
isodate               0.7.2
Jinja2                3.1.6
joblib                1.4.2
kiwisolver            1.4.7
MarkupSafe            2.1.5
matplotlib            3.7.5
networkx              3.1
numpy                 1.24.4
packaging             25.0
pandas                2.0.3
pillow                10.4.0
pip                   24.3.1
protobuf              5.29.5
psutil                7.0.0
pyparsing             3.1.4
python-dateutil       2.9.0.post0
python-louvain        0.16
pytz                  2025.2
PyYAML                6.0.2
rdflib                7.1.4
regex                 2024.11.6
requests              2.32.4
scikit-learn          1.3.2
scipy                 1.10.1
seaborn               0.13.2
setuptools            75.3.0
six                   1.17.0
tensorboardX          2.6.2.2
threadpoolctl         3.5.0
torch                 1.12.1+cu116
torch-cluster         1.6.0+pt112cu116
torch_geometric       2.4.0
torch-scatter         2.1.0+pt112cu116
torch-sparse          0.6.16+pt112cu116
torch-spline-conv     1.2.1+pt112cu116
torchaudio            0.12.1+cu116
torchvision           0.13.1+cu116
tqdm                  4.67.1
typing_extensions     4.13.2
tzdata                2025.2
urllib3               2.2.3
wcwidth               0.2.13
wheel                 0.45.1
yacs                  0.1.8
zipp                  3.20.2
```

## Training
### 1. Run the model on IEMOCAP dataset:
```bash
python -u train.py --lr 0.0001 --batch-size 16 --epochs 150 --temp 1 --Dataset 'IEMOCAP' --int_gamma 0.00434 --topk 0.7 --seed 1479
### 2. Run the model on MELD dataset:
```bash
python -u train.py --lr 0.000005 --batch-size 8 --epochs 50 --temp 8 --Dataset 'MELD' --int_gamma 0.056 --topk 0.96 --seed 2094
```
