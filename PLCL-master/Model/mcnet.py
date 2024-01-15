import torch
from torch import nn
from Model.networks_other import init_weights
import torch.nn.functional as F

filters = [16, 32, 64, 128, 256]

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, model_type= 0):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        if model_type == 0:
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        elif model_type == 1:
            self.up = nn.ConvTranspose3d(in_size, in_size, kernel_size=2, padding=0, stride=2)
        elif model_type == 2:
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode="nearest")
        else: 
            print('error')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=True, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        self.is_batchnorm = normalization
        # downsampling
        self.conv1 = UnetConv3(n_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 =    nn.MaxPool3d(kernel_size=(2, 2, 2)) # DownsamplingConvBlock(filters[0], filters[0], normalization=normalization)

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 =   nn.MaxPool3d(kernel_size=(2, 2, 2)) # DownsamplingConvBlock(filters[1], filters[1], normalization=normalization)

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 =   nn.MaxPool3d(kernel_size=(2, 2, 2)) # DownsamplingConvBlock(filters[2], filters[2], normalization=normalization)

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 =   nn.MaxPool3d(kernel_size=(2, 2, 2)) # DownsamplingConvBlock(filters[3], filters[3], normalization=normalization)

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.dropout = nn.Dropout3d(p=0.3)
        

    def forward(self, input):
        x1 = self.conv1(input)
        maxpool1 = self.maxpool1(x1)

        x2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(x2)

        x3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(x3)

        x4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(x4)

        x5 = self.center(maxpool4)
        

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res

class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_residual=False, up_type=0):
        super(Decoder, self).__init__()

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm=True, model_type=up_type)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm=True, model_type=up_type)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm=True, model_type=up_type)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm=True, model_type=up_type)

        # self.dropout1 = nn.Dropout3d(p=0.3)# 0.5
        self.dropout2 = nn.Dropout3d(p=0.3)
        self.dropout3 = nn.Dropout3d(p=0.2)
        self.dropout4 = nn.Dropout3d(p=0.1)
        self.outconv = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        up4 = self.up_concat4(x4, x5)
        # up4 = self.dropout1(up4)

        up3 = self.up_concat3(x3, up4)
        # up3 = self.dropout2(up3)

        up2 = self.up_concat2(x2, up3)
        # up2 = self.dropout3(up2)

        up1 = self.up_concat1(x1, up2)
        # up1 = self.dropout4(up1)
        up1 = self.dropout2(up1)

        out = self.outconv(up1)
        return out

class MCNet3d(nn.Module):

    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_residual=False):
        super(MCNet3d, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_residual, 1)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        return out_seg1, out_seg2
    
class MCNet3d_v1(nn.Module):

    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_residual=False):
        super(MCNet3d_v1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_residual, 1)
        self.decoder3 = Decoder(n_channels, n_classes, n_filters, normalization, has_residual, 2)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        out_seg3 = self.decoder3(features)

        return out_seg1, out_seg2, out_seg3
    
class unet_3D(nn.Module):

    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='none', has_residual=False):
        super(unet_3D, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_residual, 0)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1