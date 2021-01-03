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
        # 1st term in Eq.1
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

class Bottleneck_NoBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_NoBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.PReLU(num_parameters=planes)
        self.relu2 = nn.PReLU(num_parameters=planes)
        self.relu3 = nn.PReLU(num_parameters=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)
        return out

class ParallelModule(nn.Module):
    """a group of ResBlocks (gray box in Fig.1(c)) and a RCF (orange box in Fig.1(c)) on parallel Upper stream and Lower stream 
    """
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(ParallelModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        # self.relu = nn.ReLU(True)
        # self.relu = nn.PReLU()
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.rcf1 = RCF(inplanes=num_channels[0], planes=num_channels[0])
        self.rcf2 = RCF(inplanes=num_channels[1], planes=num_channels[1])

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride
                )
            )
        # 若需要downsample则引入1x1的卷积，否则为none
        # 首先引入衔接的block，block包含downsample(可能为none)
        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        # 根据设定的该支路block数目，stack相应数目的blocks
        # 于是，该branch的网络结构便定义好了
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        # 根据分叉的数目，调用相应次数的_make_one_branch，来构造出所有支路（每个支路包括一个衔接block，和后续的stacked blocks。衔接block可能包含一个卷积downsample）
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1): # 第一层循环是输出branch
            fuse_layer = []
            for j in range(num_branches):#第二层循环是输入branch
                if j > i: # 若输入低分辨率 -> 输出高分辨率，则先1x1卷积再nearest upsample
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0
                            ),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i: # 若输入分辨率 == 输出分辨率，则不进行任何操作
                    fuse_layer.append(None)
                else: # 若输入高分辨率 -> 输出低分辨率，则通过3x3卷积stride=2
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1
                                    )
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1
                                    ),
                                    # nn.ReLU(True)
                                    nn.PReLU(num_parameters=num_outchannels_conv3x3)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        # print(fuse_layers)
        # print('  ')
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)): # 拿到每条输出支路
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0]) #y为最高分辨率x[0]到当前输出的特征。
            # 若输出支路为最高分辨率，则直接拿来；否则由最高分辨率卷积下采样得到
            # print(y.shape)
            for j in range(1, self.num_branches):
                if i == j: # 若输出支路为i=1，
                    # y = y + x[j]
                    y = self.rcf2(backbone=x[j], bypass=y)
                else: # 若输出支路为i=0最高分辨率
                    # y = y + self.fuse_layers[i][j](x[j])
                    y = self.rcf1(backbone=y, bypass=self.fuse_layers[i][j](x[j]))
            x_fuse.append(self.relu(y))
        # print('---')
        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock_NoBN,
    'BOTTLENECK': Bottleneck_NoBN
}


class Spatial_DualStream_Module(nn.Module):
    def __init__(self, trans_in_channel=32):
        super(Spatial_DualStream_Module, self).__init__()
        # self.inplanes = inpalanes
        self.trans_in_channel = trans_in_channel # 32

        # 2
        num_channels = [32, 32]
        block = blocks_dict['BASIC']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([self.trans_in_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            num_modules=1,num_branches=2,num_blocks=[4,4],num_channels=num_channels,block=block,fuse_method='SUM', num_inchannels=num_channels,
            multi_scale_output=True)

        # 3
        num_channels = [32, 32]
        block = blocks_dict['BASIC']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            num_modules=1,num_branches=2,num_blocks=[4, 4],num_channels=num_channels,block=block,fuse_method='SUM', num_inchannels=num_channels,
            multi_scale_output=True)

        # 4
        num_channels = [32, 32]
        block = blocks_dict['BASIC']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            num_modules=1,num_branches=2,num_blocks=[4, 4],num_channels=num_channels,block=block,fuse_method='SUM', num_inchannels=num_channels,
            multi_scale_output=False)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur): #遍历每个branch
            if i < num_branches_pre:
                # 无需下采样的支路
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    # 若通道数改变，则使用卷积改变通道数
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1
                            ),
                            # nn.ReLU(inplace=True)
                            nn.PReLU(num_parameters=num_channels_cur_layer[i])
                        )
                    )
                else:
                    # 若通道数不变，则不做处理
                    transition_layers.append(None)
            else: #i > num_branches_pre
                # 需要下采样的支路
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1
                            ),
                            # nn.ReLU(inplace=True)
                            nn.PReLU(num_parameters=outchannels)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    
    def _make_stage(self, num_modules,num_branches,num_blocks,num_channels,block,fuse_method, 
                    num_inchannels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                ParallelModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)


        x_list = []
        for i in range(2):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage3(x_list)

        x_list = []
        for i in range(2):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage4(x_list)

        return y_list[0]



class Spatial_Stage(nn.Module):
    def __init__(self, ):
        super(Spatial_Stage, self).__init__()
        # 原本是3->64->32的两层3x3卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)# orig stride=3
        self.resbk1 = BasicBlock_NoBN(inplanes=64, planes=64, stride=1, downsample=None) #内部包含了两次prelu
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True) # channel reduction
        self.resbk2 = BasicBlock_NoBN(inplanes=32, planes=32, stride=1, downsample=None)                            
        
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.PReLU(num_parameters=64)
        self.relu2 = nn.PReLU(num_parameters=32)
        
        self.spatial_dualstream_module1 = Spatial_DualStream_Module()
        self.spatial_dualstream_module2 = Spatial_DualStream_Module()

        self.cat_conv = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu_cat = nn.PReLU(num_parameters=32)

        self.ps_conv = nn.Conv2d(32, 32*4, kernel_size=3, stride=1, padding=1)

        self.conv_sr1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_sr2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=True)
        # self.relu_sr1 = nn.ReLU(inplace=True)
        self.relu_sr1 = nn.PReLU(num_parameters=32)
        # self.relu_sr2 = nn.ReLU(inplace=True)
        self.relu_sr2 = nn.PReLU(num_parameters=32)
        self.conv_final = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, input):

        x = self.relu1(self.conv1(input))
        x = self.resbk1(x)
        x = self.relu2(self.conv2(x))
        x = self.resbk2(x)

        tmp0 = self.spatial_dualstream_module1(x) + x
        tmp1 = self.spatial_dualstream_module2(tmp0) + tmp0

        y = self.relu_cat(self.cat_conv(torch.cat([tmp0, tmp1], dim=1))) # cat of two features, and map from 64-channel to 32-channel

        y_upsampled = torch.nn.functional.pixel_shuffle(self.ps_conv(y), upscale_factor=2)

        output = self.conv_final(
            self.relu_sr2(self.conv_sr2(
                self.relu_sr1(self.conv_sr1(y_upsampled))
            ))) \
            + torch.nn.functional.interpolate(input, scale_factor=2, mode='nearest')

        return output

    def weight_init(self, mode='kaiming'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init
        elif mode == 'kaiming_uniform':
            initializer = kaiming_uniform_init
        elif mode == 'xavier_uniform':
            initializer = xavier_uniform_init
        for m in self.modules():
            initializer(m)


# init weights function
def xavier_uniform_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def kaiming_uniform_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)



if __name__ == '__main__':
    
    model = Spatial_Stage()
    x = torch.autograd.Variable(torch.FloatTensor(8,3,96,96))
    y = model(x)
