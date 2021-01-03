import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from math import sqrt

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init2(m[-1], val=0)
    else:
        constant_init2(m, val=0)

def kaiming_init2(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init2(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class RCF(nn.Module):
    def __init__(self, inplanes, planes, pool='att', fusions=['channel_add'], ratio=2):
        super(RCF, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        self.se_backone = SELayer(channel=inplanes) # 1st term in Eq.1
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)#context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                # nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                # nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init2(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, backbone, bypass):
        batch, channel, height, width = backbone.size()
        if self.pool == 'att':
            input_x = bypass
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(backbone)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)#softmax操作
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(bypass)

        return context

    def forward(self, backbone, bypass):
        # [N, C, 1, 1]
        context = self.spatial_pool(backbone, bypass)
        backbone = self.se_backone(backbone) #两支路合并之前，在当前支路上添加一个SE模块, 20200226
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            # out = x * channel_mul_term
            out = backbone * channel_mul_term
        else:
            # out = x
            out = backbone
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            # out = out + channel_add_term
            backbone = backbone + channel_add_term
        return out

class WFA(nn.Module):
    """
    Wavelet Feature Adaptation
    """
    def __init__(self, channel, reduction=2):
        super(WFA, self).__init__()
        self.tmp_chnl_num = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, self.tmp_chnl_num, bias=True),
            nn.PReLU(num_parameters=self.tmp_chnl_num),
            nn.Linear(self.tmp_chnl_num, channel, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1) # scale factor: attention mapping vector
        return x * y

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True) # Once remove BN, turn the no-bias into bias

class BasicBlock_NoBN(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_NoBN, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.PReLU(num_parameters=planes)
        self.relu2 = nn.PReLU(num_parameters=planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        return out


class WAC(nn.Module):
    """Wavelet-Aware Convolutonal block
    """
    def __init__(self, inplanes, planes, stride=1):
        super(WAC, self).__init__()
        self.l_conv = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=stride,
                    padding=1, bias=True)
        self.l_relu = nn.PReLU(num_parameters=planes)

        self.h_conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 3), stride=stride,
                    padding=(0,1), bias=True)
        self.h_relu1 = nn.PReLU(num_parameters=planes)
        self.h_conv2 = nn.Conv2d(planes, planes, kernel_size=(1, 3), stride=stride,
                    padding=(0,1), bias=True)
        self.h_relu2 = nn.PReLU(num_parameters=planes)
        
        self.v_conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 1), stride=stride,
                    padding=(1,0), bias=True)
        self.v_relu1 = nn.PReLU(num_parameters=planes)
        self.v_conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 1), stride=stride,
                    padding=(1,0), bias=True)
        self.v_relu2 = nn.PReLU(num_parameters=planes)

        self.d_conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=stride,
                    padding=1, bias=True)
        self.d_relu1 = nn.PReLU(num_parameters=planes)
        self.d_conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride,
                    padding=1, bias=True)
        self.d_relu2 = nn.PReLU(num_parameters=planes)

    def forward(self, x):
        residual = x
        
        out_l = self.l_relu(self.l_conv(x))
        out_h = self.h_relu2(self.h_conv2(self.h_relu1(self.h_conv1(x))))
        out_v = self.v_relu2(self.v_conv2(self.v_relu1(self.v_conv1(x))))
        out_d = self.d_relu2(self.d_conv2(self.d_relu1(self.d_conv1(x))))

        # out = residual + out_l + out_h + out_v +out_d
        out = out_l + out_h + out_v +out_d
        return out

def dwt_init(x):
    """wavelet packet transform, as illustrated in Eq.2"""
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    """reverse wavelet packet transform"""
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class Wavelet_Stage(nn.Module):
    """
        chnl_list: the 1-level basis contains 12 channels (96*96)
                   the 2-level basis contains 48 channels (48*48)

    """
    def __init__(self, decoder_chnls_list=[24, 24], encoder_up_chnls_list=[24, 24, 24], encoder_down_chnls_list=[96, 96, 96]):
        super(Wavelet_Stage, self).__init__()

        self.wfa_up = WFA(12, reduction=2)        
        self.wfa_down = WFA(48, reduction=2)
        self.iwfa_up = WFA(12, reduction=2)
        
        self.rcf_up_1 = RCF(inplanes=decoder_chnls_list[0], planes=decoder_chnls_list[0])
        self.rcf_down_1 = RCF(inplanes=4*decoder_chnls_list[0], planes=4*decoder_chnls_list[0])
        self.rcf_up_2 = RCF(inplanes=decoder_chnls_list[1], planes=decoder_chnls_list[1])
        self.rcf_down_2 = RCF(inplanes=4*decoder_chnls_list[1], planes=4*decoder_chnls_list[1])
        self.rcf_up_3 = RCF(inplanes=12, planes=12)

        self.conv_final = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, bias=True)

        self.wconv_up_encoder1 = WAC(inplanes=12, planes=encoder_up_chnls_list[0], stride=1)
        self.wconv_up_decoder1 = WAC(inplanes=encoder_up_chnls_list[0], planes=decoder_chnls_list[0], stride=1)
        self.wconv_down_encoder1 = WAC(inplanes=48, planes=encoder_down_chnls_list[0], stride=1)
        self.wconv_down_decoder1 = WAC(inplanes=encoder_down_chnls_list[0], planes=4*decoder_chnls_list[0], stride=1)
        ###
        self.wconv_up_encoder2 = WAC(inplanes=decoder_chnls_list[0], planes=encoder_up_chnls_list[1], stride=1)
        self.wconv_up_decoder2 = WAC(inplanes=encoder_up_chnls_list[1], planes=decoder_chnls_list[1], stride=1)
        self.wconv_down_encoder2 = WAC(inplanes=4*decoder_chnls_list[0], planes=encoder_down_chnls_list[1], stride=1)
        self.wconv_down_decoder2 = WAC(inplanes=encoder_down_chnls_list[1], planes=4*decoder_chnls_list[1], stride=1)
        ###
        self.wconv_up_encoder3 = WAC(inplanes=decoder_chnls_list[1], planes=encoder_up_chnls_list[2], stride=1)
        self.wconv_up_decoder3 = WAC(inplanes=encoder_up_chnls_list[2], planes=12, stride=1) # to the final IWPT

        #
        self.wconv_down_encoder3 = WAC(inplanes=4*decoder_chnls_list[1], planes=encoder_down_chnls_list[2], stride=1)
        self.wconv_down_decoder3 = WAC(inplanes=encoder_down_chnls_list[2], planes=48, stride=1) # 

    def forward(self, x):
        x_w_up = dwt_init(x)
        x_w_down = dwt_init(x_w_up)
        x_w_up_adapted = self.wfa_up(x_w_up)
        x_w_down_adapted = self.wfa_down(x_w_down)

        out_up_1 = self.wconv_up_decoder1(self.wconv_up_encoder1(x_w_up_adapted))
        out_down_1 = self.wconv_down_decoder1(self.wconv_down_encoder1(x_w_down_adapted))
        
        out_up_2 = self.wconv_up_decoder2(self.wconv_up_encoder2(self.rcf_up_1(backbone=out_up_1, bypass=iwt_init(out_down_1))))
        out_down_2 = self.wconv_down_decoder2(self.wconv_down_encoder2(self.rcf_down_1(backbone=out_down_1, bypass=dwt_init(out_up_1))))

        out_up_3 = self.wconv_up_decoder3(self.wconv_up_encoder3(self.rcf_up_2(backbone=out_up_2, bypass=iwt_init(out_down_2))))
        out_down_3 = self.wconv_down_decoder3(self.wconv_down_encoder3(self.rcf_down_2(backbone=out_down_2, bypass=dwt_init(out_up_2))))

        wave_predict = self.conv_final(self.rcf_up_3(backbone=out_up_3, bypass=iwt_init(out_down_3))) + x_w_up
        y = iwt_init(wave_predict)

        return y, wave_predict



if __name__ == '__main__':
    
    model = Wavelet_Stage()
    input_tensor = torch.FloatTensor(4,3,32,32)
    var = torch.autograd.Variable(input_tensor)

    print(model(var))

