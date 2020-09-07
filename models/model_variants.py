import torch.nn as nn
import functools
import torch
import functools
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision

class Bottleneck(nn.Module):

    # expansion = 4
    def __init__(self, inplanes, outplanes, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, int(inplanes/4), kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(int(inplanes/4))
        self.conv2 = nn.Conv2d(int(inplanes/4), int(inplanes/4), kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(int(inplanes/4))
        self.conv3 = nn.Conv2d(int(inplanes/4), outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(outplanes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class ASBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False):
        super(ASBlock, self).__init__()
        self.conv_block_stream1 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=True, cated_stream2=cated_stream2)
        
        self.channel_switch = nn.Conv2d(dim * 2, dim, kernel_size=1, padding=0, bias=False)
        self.channel_switch_N = nn.InstanceNorm2d(dim)
        self.channel_switch_A = nn.LeakyReLU(True)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False, cal_att=False):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if cated_stream2:
            conv_block += [nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim*2),
                       nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim),
                           nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if cal_att:
            if cated_stream2:
                conv_block += [nn.Conv2d(dim*2, dim, kernel_size=3, padding=p, bias=use_bias)]
            else:
                conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x1, x2):

        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        att = F.sigmoid(x2_out)
        x1_out = torch.cat([x1_out ,att],1)
        x1_out = self.channel_switch(x1_out)
        x1_out = self.channel_switch_N(x1_out)
        x1_out_after = self.channel_switch_A(x1_out)
        out = x1 + x1_out_after # residual connection
        # stream2 receive feedback from stream1
        x2_out = torch.cat((x2_out, out), 1)
        return out, x2_out, x1, x1_out_after

class ASNModel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=2):
        assert(n_blocks >= 0 and type(input_nc) == list)
        super(ASNModel, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        

        self.model_stream1_down_Reflect = nn.ReflectionPad2d(3)
        self.model_stream1_down_Con1 = nn.Conv2d(self.input_nc_s1, 64, kernel_size=7, padding=0, bias=False)
        self.model_stream1_down_N1 = nn.InstanceNorm2d(64)
        self.model_stream1_down_A1 = nn.LeakyReLU(True)

        self.model_stream2_down_Reflect = nn.ReflectionPad2d(3)
        self.model_stream2_down_Con1 = nn.Conv2d(self.input_nc_s2, 64, kernel_size=7, padding=0,
                           bias=False)
        self.model_stream2_down_N1 = nn.InstanceNorm2d(64)
        self.model_stream2_down_A1 = nn.LeakyReLU(True)

        self.model_stream1_down_Con2 = nn.Conv2d(64 ,128, kernel_size=3,
                                stride=2, padding=1, bias=False)
        self.model_stream1_down_N2 = nn.InstanceNorm2d(128)
        self.model_stream1_down_A2 = nn.LeakyReLU(True)

        self.model_stream2_down_Con2 = nn.Conv2d(64 , 128, kernel_size=3,
                                stride=2, padding=1, bias=False)
        self.model_stream2_down_N2 = nn.InstanceNorm2d(128)
        self.model_stream2_down_A2 = nn.LeakyReLU(True)

        self.model_stream1_down_Con3 = nn.Conv2d(128, 256, kernel_size=3,
                                stride=2, padding=1, bias=False)
        self.model_stream1_down_N3 = nn.InstanceNorm2d(256)
        self.model_stream1_down_A3 = nn.LeakyReLU(True)

        self.model_stream2_down_Con3 = nn.Conv2d(128, 256, kernel_size=3,
                                stride=2, padding=1, bias=False)
        self.model_stream2_down_N3 = nn.InstanceNorm2d(256)
        self.model_stream2_down_A3 = nn.LeakyReLU(True)

        self.model_stream1_down_Con4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.model_stream1_down_N4 = nn.InstanceNorm2d(512)
        self.model_stream1_down_A4 = nn.LeakyReLU(True)

        self.model_stream2_down_Con4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.model_stream2_down_N4 = nn.InstanceNorm2d(512)
        self.model_stream2_down_A4 = nn.LeakyReLU(True)

        cated_stream2 = [True for i in range(4)]
        cated_stream2[0] = False
        asBlock = nn.ModuleList()
        for i in range(4):
            asBlock.append(ASBlock(512, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=False, cated_stream2=cated_stream2[i]))
  
        self.layer0 = self._make_layer(2, 1024, 1024)

        self.model_stream1_up_Con0_rgb = nn.ConvTranspose2d(1024, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.model_stream1_up_A0_rgb = nn.Tanh()

        self.model_stream1_up_Con0 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.model_stream1_up_N0 = nn.InstanceNorm2d(512)
        self.model_stream1_up_A0 = nn.ReLU(True)

        self.layer1 = self._make_layer(2, 771, 771)

        self.model_stream1_up_Con1_rgb = nn.ConvTranspose2d(771, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.model_stream1_up_A1_rgb = nn.Tanh()

        self.model_stream1_up_Con1 = nn.ConvTranspose2d(771, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.model_stream1_up_N1 = nn.InstanceNorm2d(256)
        self.model_stream1_up_A1 = nn.ReLU(True)

        self.layer2 = self._make_layer(2, 387, 387)

        self.model_stream1_up_Con2_rgb = nn.ConvTranspose2d(387, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.model_stream1_up_A1_rgb = nn.Tanh()

        self.model_stream1_up_Con2 = nn.ConvTranspose2d(387 , 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.model_stream1_up_N2 = nn.InstanceNorm2d(128)
        self.model_stream1_up_A2 = nn.ReLU(True)

        self.model_stream1_up_Reflect = nn.ReflectionPad2d(1)
        self.model_stream1_up_Con3 = nn.Conv2d(128 , 3, kernel_size=3, padding=0, bias=False)
        self.model_stream1_up_A3 = nn.Tanh()

        self.model_stream1_up_Con5 = nn.Conv2d(6 , 3, kernel_size=1, padding=0, bias=False)
        self.model_stream1_up_A5 = nn.Tanh()

        self.asBlock = asBlock

    def _make_layer(self, block, planes, outplanes):

        layers = []
        layers.append(Bottleneck(planes, outplanes))
        for i in range(1, block):
            layers.append(Bottleneck(outplanes, outplanes))

        return nn.Sequential(*layers)

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, input):

        x1, x2 = input

        # eye, nose, mouth are for TP-GAN

        # Down Sampling
        x1 = self.model_stream1_down_Reflect(x1)
        x1 = self.model_stream1_down_Con1(x1)
        x1 = self.model_stream1_down_N1(x1)
        x1 = self.model_stream1_down_A1(x1)

        x2 = self.model_stream2_down_Reflect(x2)
        x2 = self.model_stream2_down_Con1(x2)
        x2 = self.model_stream2_down_N1(x2)
        x2 = self.model_stream2_down_A1(x2)

        x1 = self.model_stream1_down_Con2(x1)
        x1 = self.model_stream1_down_N2(x1)
        x1 = self.model_stream1_down_A2(x1)
    
        x2 = self.model_stream2_down_Con2(x2)
        x2 = self.model_stream2_down_N2(x2)
        x2 = self.model_stream2_down_A2(x2)

        x_64 = x1

        x1 = self.model_stream1_down_Con3(x1)
        x1 = self.model_stream1_down_N3(x1)
        x1 = self.model_stream1_down_A3(x1)
        
        x2 = self.model_stream2_down_Con3(x2)
        x2 = self.model_stream2_down_N3(x2)
        x2 = self.model_stream2_down_A3(x2)

        x_32 = x1

        x1 = self.model_stream1_down_Con4(x1)
        x1 = self.model_stream1_down_N4(x1)
        x1 = self.model_stream1_down_A4(x1)

        x_16 = x1

    
        x2 = self.model_stream2_down_Con4(x2)
        x2 = self.model_stream2_down_N4(x2)
        x2 = self.model_stream2_down_A4(x2)

        # AS-Block

        att = torch.sigmoid(x2)
        x1_out = x1 * att
        x1 = x1 + x1_out 
        before_list = []
        after_list =[]
        for model in self.asBlock:
            x1, x2, x1_before, x1_after = model(x1, x2)
            before_list.append(x1_before)
            after_list.append(x1_after)

        x1 = torch.cat([x1 ,x_16],1)

        x1 = self.layer0(x1)

        fake_16 = self.model_stream1_up_Con0_rgb(x1)
        fake_16 = self.model_stream1_up_A0_rgb(fake_16)
        fake_16_32 = torch.nn.functional.upsample(fake_16,(32,32),mode='bilinear')

        x1 = self.model_stream1_up_Con0(x1)
        x1 = self.model_stream1_up_N0(x1)
        x1 = self.model_stream1_up_A0(x1)

        x1 = torch.cat([x1 ,x_32],1)

        x1 = torch.cat([x1 ,fake_16_32],1)

        x1 = self.layer1(x1)

        fake_32 = self.model_stream1_up_Con1_rgb(x1)
        fake_32 = self.model_stream1_up_A0_rgb(fake_32)
        fake_32_64 = torch.nn.functional.upsample(fake_32,(64,64),mode='bilinear')

        x1 = self.model_stream1_up_Con1(x1)
        x1 = self.model_stream1_up_N1(x1)
        x1 = self.model_stream1_up_A1(x1)

        x1 = torch.cat([x1 ,x_64],1)

        x1 = torch.cat([x1 ,fake_32_64],1)

        x1 = self.layer2(x1)

        fake_64 = self.model_stream1_up_Con2_rgb(x1)
        fake_64 = self.model_stream1_up_A0_rgb(fake_64)
        fake_64_128 = torch.nn.functional.upsample(fake_64,(128,128),mode='bilinear')

        x1 = self.model_stream1_up_Con2(x1)
        x1 = self.model_stream1_up_N2(x1)
        x1 = self.model_stream1_up_A2(x1)

        x1 = self.model_stream1_up_Reflect(x1)
        x1 = self.model_stream1_up_Con3(x1)
        x1 = self.model_stream1_up_A3(x1)

        x1 = torch.cat([x1 ,fake_64_128],1)

        x1 = self.model_stream1_up_Con5(x1)
        x1 = self.model_stream1_up_A5(x1)

        return x1, fake_64, fake_32, fake_16, before_list, after_list



class ASNetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=2):
        super(ASNetwork, self).__init__()
        assert type(input_nc) == list and len(input_nc) == 2, 'The AttModule take input_nc in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = ASNModel(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids, padding_type, n_downsampling=n_downsampling)

    def forward(self, input):
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)






