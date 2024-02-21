# A machine learning approach to robustly determine director fields and analyze defects in active nematics
## Introduction
This project trained a machine learning model to robustly extract director fields from raw experimental images. It is less sensitive to noises compared to traditional computer vision methods. Using the model, the calcualted director fields enabled more accurate downstream tasks such as defect detection, analysis and tracking. 
## Requirements and Installation
### 1. Create virtual environment
```
conda create -n orientationfinder python=3.9 
conda activate orientationfinder
```

### 2. Install pytorch with CUDA 11.7
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 
pip install pytorch_lightning 
pip install pandas 
pip install nmrglue 
pip install transformers
```
## Train and Evaluation 
Training is done by running the `train_gabor_resnet_aug_smoothed.py` file.

To run an experiment, the following parameters can be specified:

--model_name: name to save model weights

--aug: if specified, training data will be augmented

--iter: total training iterations

--batch: batch sizes

--image_size: input image size

--ckpt: path to the checkpoint if resume training

--lr: learning rate

--lr_step: learning rate change after certain step

--lr_gamma: learning rate gamma for rate change

--train_gabor: whether to train gabor filter or not

--channels: channels for the resnet model

--layers: layers for the resnet model

--scale: scale for gabor filter

--orientation: orientation for gabor filter

--lam: wave length of gabor filter

--kernel: kernel size for gabor filter


Evaluation can be done by running `eval_model_multidata.py` file.
  
## Code Description
`ansim_dataset_unconf.py`: dataloader file
`model.py` contains all module needed for the orientatio finder model. 
`simmim.py` uses the masking strategy from [1]. The masking strategy is used to correct the unsmoothness of the raw experimental images to obtain a more accurate training target. 
`nematic_plot.py` is used to visualize the director fields calculated. 

## Citation 

If you use this code or its corresponding [paper]([https://pubs.rsc.org/en/content/articlehtml/2024/sm/d3sm01253k]), please cite our work as follows:

```
@article{li2024machine,
  title={A machine learning approach to robustly determine director fields and analyze defects in active nematics},
  author={Li, Yunrui and Zarei, Zahra and Tran, Phu and Wang, Yifei and Hong, Pengyu and Baskaran, Aparna and Hagan, Michael F and Fraden, Seth},
  journal={Soft matter},
  year={2024},
  publisher={Royal Society of Chemistry}
}
```

## Reference
[1] Xie, Zhenda, et al. "Simmim: A simple framework for masked image modeling." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
