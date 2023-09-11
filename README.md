# ARC-Depth

This repo is for **Devil's in the Reconstruction: Self-Supervised Monocular Depth Estimation from Videos via Adaptive Reconstruction Constraints**

## Setup
Assuming a fresh Anaconda distribution, you can install the dependencies with:
```
pip3 install torch==1.3.0 torchvision==0.4.1
pip install numpy pandas matplotlib scikit-image scipy imageio tqdm cython yacs pypng mmcv==0.4.4 Pillow==6.2.2
```

<!-- ## Comparing with others
![](images/table1.png) -->

## Method
<p align="center">
  <img src="images/overview.png" alt="overviewpng" width="800" />
</p>
<p align="center">Overview of our proposed network</p>

## Training:

for student net training, please check the sh file for different experiment settings and run:
```
sh start2train_student.sh
```


for teacher net training, please run:
```
sh start2train_teacher.sh
```

## Testing:

for student net testing, please run:
```
sh disp_evaluation_student.sh
```


for teacher net testing, please run:
```
sh disp_evaluation_teacher.sh
```

## Infer depth maps from RGB images in a folder:

```
sh test_sample.sh
```

#### Acknowledgement
 Thanks the authors for their works:
 - [monodepth2](https://github.com/nianticlabs/monodepth2)
 - [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)
 - [FeatDepth](https://github.com/sconlyshootery/FeatDepth)
 - [DIFFNet](https://github.com/brandleyzhou/DIFFNet)

<!-- ## Citation

If this codebase or our method helps your research, please cite: -->
