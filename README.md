# SWD-Net

This repository is an official PyTorch implementation of the paper **"Joint Spatial-Wavelet Dual-Stream Network for Super-Resolution"** [[paper](https://www.researchgate.net/publication/346066209_Joint_Spatial-Wavelet_Dual-Stream_Network_for_Super-Resolution)] from **MICCAI 2020**.

**HistoSR** dataset with bicubic and nearest degradation is public from [here](alink).

![<div align=center><img width="150" height="150">](/figs/framework.png)

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
