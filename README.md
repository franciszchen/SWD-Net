# SWD-Net

This repository is an official PyTorch implementation of the paper **"Joint Spatial-Wavelet Dual-Stream Network for Super-Resolution"** [[paper](https://www.researchgate.net/publication/346066209_Joint_Spatial-Wavelet_Dual-Stream_Network_for_Super-Resolution)] from **MICCAI 2020**.

<div align=center><img width="500" src=/figs/framework.png></div>

## HistoSR
**HistoSR** dataset is built by random cropping patches from Camelyon16 dataset. By bicubic and nearest downsampling, HistoSR dataset provides a 2 \times SR from 96 \times 96 pixels to 192 \times 192 pixels with two kinds of degradation. Specifically, the bicubic degradation kernel is the common choice and retains neighboring information, while the nearest one discards the pixels directly. In this way, the nearest version provides a more difficult case to comprehensively evaluate various SR algorithms. Each version of HistoSR dataset contains 30,000 SR pairs in training set and 5,000 SR pairs in test set.

with bicubic and nearest degradation is public from [here](alink).


## Dependencies
* Python 3.6
* PyTorch >= 1.3.0
* numpy
* lmdb

## Code
Clone this repository into any place you want.
```bash
git clone https://github.com/franciszchen/SWD-Net.git
cd SWD-Net
```

## Quickstart 

## Cite
If you find our work useful in your research or publication, please cite our work:
```
@inproceedings{chen2020joint,
  title={Joint Spatial-Wavelet Dual-Stream Network for Super-Resolution},
  author={Chen, Zhen and Guo, Xiaoqing and Yang, Chen and Ibragimov, Bulat and Yuan, Yixuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={184--193},
  year={2020},
  organization={Springer}
}
```

## Acknowledgements
* SSIM calculation derived from [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim).
* Wavelet Packet Transform derived from [MWCNN_PyTorch](https://github.com/lpj0/MWCNN_PyTorch).
