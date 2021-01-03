import torch
import torch.nn as nn

class SWDNet(nn.Module):
    def __init__(self, spatial_sr, wavelet_sr):
        super(SWDNet, self).__init__()
        self.add_module('spatial_sr', spatial_sr)
        self.add_module('wavelet_sr', wavelet_sr)

    def forward(self, x):
        sr_spatial = self.spatial_sr(x)
        sr_wavelet, output_wavelet = self.wavelet_sr(sr_spatial)
        return sr_wavelet, output_wavelet, sr_spatial


if __name__ == '__main__':
    import spatial_stage
    import wavelet_stage

    spatial_sr = spatial_stage.Spatial_Stage()
    wavelet_sr = wavelet_stage.Wavelet_Stage()

    swdnet = SWDNet(spatial_sr, wavelet_sr)
    print(swdnet)

    