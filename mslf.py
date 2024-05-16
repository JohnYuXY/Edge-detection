import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from math import sqrt

class selfattention(nn.Module):
    def __init__(self,input_dim,dim_k,dim_v):
        super(selfattention,self).__init__()
        self.q=nn.Linear(input_dim,dim_k)
        self.k=nn.Linear(input_dim,dim_k)
        self.v=nn.Linear(input_dim,dim_v)
        self.normfact=1/sqrt(dim_k)

    def forward(self,x):
        Q=self.q(x)
        K=self.k(x)
        V=self.v(x)
        atten=nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1)))*self.normfact
        output=torch.bmm(atten,V)
        return output

class MSLF(nn.Module):
    def __init__(self, pretrained=None):
        super(MSLF, self).__init__()
        self.bn1=nn.BatchNorm2d(64)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(256)
        self.bn4=nn.BatchNorm2d(512)

        self.conv_sp = nn.Conv2d( 64, 128, 1, stride=2,padding=0, dilation=1)
        self.conv_sp2 = nn.Conv2d( 128, 256, 1, stride=2,padding=0, dilation=1)
        self.conv_sp3 = nn.Conv2d( 256, 512, 1, stride=2,padding=0, dilation=1)

        self.conv2_1x1 = nn.Conv2d(64, 16, 1, stride=1, padding=0, dilation=1)
        self.conv2_3x3 = nn.Conv2d(64, 16, 3, stride=1, padding=1, dilation=1)
        self.conv2_5x5 = nn.Conv2d(64, 16, 5, stride=1, padding=2, dilation=1)
        self.conv2de = nn.ConvTranspose2d(64, 16, 3, stride=1, padding=0, dilation=1)
        self.conv2de_con = nn.Conv2d(16, 16, 3, stride=1, padding=0, dilation=1)

        self.conv3_1x1 = nn.Conv2d(64, 32, 1, stride=2, padding=0, dilation=1)
        self.conv3_3x3 = nn.Conv2d(64, 32, 3, stride=2, padding=1, dilation=1)
        self.conv3_5x5 = nn.Conv2d(64, 32, 5, stride=2, padding=2, dilation=1)
        self.conv3de = nn.ConvTranspose2d(64, 32, 3, stride=1, padding=0, dilation=1)
        self.conv3de_con = nn.Conv2d(32, 32, 3, stride=2, padding=0, dilation=1)

        self.conv4_1x1 = nn.Conv2d(128, 64, 1, stride=2, padding=0, dilation=1)
        self.conv4_3x3 = nn.Conv2d(128, 64, 3, stride=2, padding=1, dilation=1)
        self.conv4_5x5 = nn.Conv2d(128, 64, 5, stride=2, padding=2, dilation=1)
        self.conv4de = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=0, dilation=1)
        self.conv4de_con = nn.Conv2d(64, 64, 3, stride=2, padding=0, dilation=1)

        self.conv5_1x1 = nn.Conv2d(256, 128, 1, stride=2, padding=0, dilation=1)
        self.conv5_3x3 = nn.Conv2d(256, 128, 3, stride=2, padding=1, dilation=1)
        self.conv5_5x5 = nn.Conv2d(256, 128, 5, stride=2, padding=2, dilation=1)
        self.conv5de = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=0, dilation=1)
        self.conv5de_con = nn.Conv2d(128, 128, 3, stride=2, padding=0, dilation=1)


        self.conv1_1 = nn.Conv2d(  3,  64, 3, stride=1,padding=1, dilation=1)
        self.conv1_2 = nn.Conv2d( 64, 64, 3, stride=1,padding=1, dilation=1)
        self.conv2_1 = nn.Conv2d( 64, 64, 3, stride=1,padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d( 64, 64, 3, stride=1,padding=1, dilation=1)
        self.conv2_3 = nn.Conv2d( 64, 64, 3, stride=1,padding=1, dilation=1)
        self.conv2_4 = nn.Conv2d( 64, 64, 3, stride=1,padding=1, dilation=1)
        self.conv2_5 = nn.Conv2d( 64, 64, 3, stride=1,padding=1, dilation=1)
        self.conv2_6 = nn.Conv2d( 64, 64, 3, stride=1,padding=1, dilation=1)
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2,padding=1, dilation=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv3_4 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv3_5 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv3_6 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv3_7 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv3_8 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv4_1 = nn.Conv2d(128, 256, 3, stride=2,padding=1, dilation=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_4 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_5 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_6 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_7 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_8 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_9 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_10= nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_11= nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_12= nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv5_1 = nn.Conv2d(256, 512, 3, stride=2,padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_5 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_6 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        # self.pool1 = nn.MaxPool2d(3,2,1)
        # self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.act = nn.ReLU(inplace=True)

        self.conv1_1_down = nn.Conv2d(64, 16, 1)
        self.conv1_2_down = nn.Conv2d(64, 16, 1)
        self.conv2_1_down = nn.Conv2d(64, 16, 1)
        self.conv2_2_down = nn.Conv2d(64, 16, 1)
        self.conv2_3_down = nn.Conv2d(64, 16, 1)
        self.conv2_4_down = nn.Conv2d(64, 16, 1)
        self.conv2_5_down = nn.Conv2d(64, 16, 1)
        self.conv2_6_down = nn.Conv2d(64, 16, 1)
        self.conv3_1_down = nn.Conv2d(128, 16, 1)
        self.conv3_2_down = nn.Conv2d(128, 16, 1)
        self.conv3_3_down = nn.Conv2d(128, 16, 1)
        self.conv3_4_down = nn.Conv2d(128, 16, 1)
        self.conv3_5_down = nn.Conv2d(128, 16, 1)
        self.conv3_6_down = nn.Conv2d(128, 16, 1)
        self.conv3_7_down = nn.Conv2d(128, 16, 1)
        self.conv3_8_down = nn.Conv2d(128, 16, 1)
        self.conv4_1_down = nn.Conv2d(256, 16, 1)
        self.conv4_2_down = nn.Conv2d(256, 16, 1)
        self.conv4_3_down = nn.Conv2d(256, 16, 1)
        self.conv4_4_down = nn.Conv2d(256, 16, 1)
        self.conv4_5_down = nn.Conv2d(256, 16, 1)
        self.conv4_6_down = nn.Conv2d(256, 16, 1)
        self.conv4_7_down = nn.Conv2d(256, 16, 1)
        self.conv4_8_down = nn.Conv2d(256, 16, 1)
        self.conv4_9_down = nn.Conv2d(256, 16, 1)
        self.conv4_10_down = nn.Conv2d(256, 16, 1)
        self.conv4_11_down = nn.Conv2d(256, 16, 1)
        self.conv4_12_down = nn.Conv2d(256, 16, 1)
        self.conv5_1_down = nn.Conv2d(512, 16, 1)
        self.conv5_2_down = nn.Conv2d(512, 16, 1)
        self.conv5_3_down = nn.Conv2d(512, 16, 1)
        self.conv5_4_down = nn.Conv2d(512, 16, 1)
        self.conv5_5_down = nn.Conv2d(512, 16, 1)
        self.conv5_6_down = nn.Conv2d(512, 16, 1)

        self.score_dsn1 = nn.Conv2d(16, 1, 1)
        self.score_dsn2 = nn.Conv2d(16, 1, 1)
        self.score_dsn3 = nn.Conv2d(16, 1, 1)
        self.score_dsn4 = nn.Conv2d(16, 1, 1)
        self.score_dsn5 = nn.Conv2d(16, 1, 1)
        self.score_fuse = nn.Conv2d(5, 1, 1)

        self.weight_deconv2 = self._make_bilinear_weights( 4, 1).cuda()
        self.weight_deconv3 = self._make_bilinear_weights( 8, 1).cuda()
        self.weight_deconv4 = self._make_bilinear_weights(16, 1).cuda()
        self.weight_deconv5 = self._make_bilinear_weights(16, 1).cuda()

        # init weights
        self.apply(self._init_weights)
        if pretrained is not None:
            checkpoint=torch.load(pretrained)
            self.load_state_dict(checkpoint['state_dict'],strict=False)
            # vgg16 = sio.loadmat(pretrained)
            # torch_params = self.state_dict()
            #
            # for k in vgg16.keys():
            #     name_par = k.split('-')
            #     size = len(name_par)
            #     if size == 2:
            #         name_space = name_par[0] + '.' + name_par[1]
            #         data = np.squeeze(vgg16[k])
            #         torch_params[name_space] = torch.from_numpy(data)
            # self.load_state_dict(torch_params)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                nn.init.constant_(m.weight, 0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Based on HED implementation @ https://github.com/xwjabc/hed
    def _make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w

    # Based on BDCN implementation @ https://github.com/pkuCactus/BDCN
    def _crop(self, data, img_h, img_w, crop_h, crop_w):
        _, _, h, w = data.size()
        # assert(img_h <= h and img_w <= w)
        data = data[:, :, crop_h:crop_h + img_h, crop_w:crop_w + img_w]
        return data

    def forward(self, x):
        device=torch.device("cuda")
        img_h, img_w = x.shape[2], x.shape[3]

        conv1_1 = self.act(self.bn1(self.conv1_1(x)))
        conv1_2 = self.act(self.bn1(self.conv1_2(conv1_1)))
        # pool1   = self.pool1(conv1_1)
        conv2_1=self.act(self.bn1(torch.cat([self.conv2_1x1(conv1_2),self.conv2_3x3(conv1_2),self.conv2_5x5(conv1_2),
                           self.conv2de_con(self.conv2de(conv1_2))],1)))
        #conv2_1 = self.act(self.bn1(self.conv2_1x1(conv1_2)))
        #引入自注意力机制
        att2_1=selfattention(conv2_1.shape[3],3,conv2_1.shape[3]).to(device)
        conv2_1=att2_1(conv2_1[0]).unsqueeze(0)

        conv2_2_add = self.bn1(self.conv2_2(conv2_1))
        conv2_2 = self.act(conv2_2_add)
        conv_next=self.act(conv1_1+conv2_2_add)

        conv2_3 = self.act(self.bn1(self.conv2_3(conv_next)))
        conv2_4_add = self.bn1(self.conv2_4(conv2_3))
        conv2_4 = self.act(conv2_4_add)
        conv_next = self.act(conv_next + conv2_4_add)
        #
        conv2_5 = self.act(self.bn1(self.conv2_5(conv_next)))
        conv2_6_add = self.bn1(self.conv2_6(conv2_5))
        conv2_6 = self.act(conv2_6_add)
        conv_next = self.act(conv_next + conv2_6_add)

        conv3_1 = self.act(self.bn2(torch.cat([self.conv3_1x1(conv_next), self.conv3_3x3(conv_next), self.conv3_5x5(conv_next),
                             self.conv3de_con(self.conv3de(conv_next))], 1)))
        #conv3_1 = self.act(self.bn2(self.conv3_1(conv_next)))
        #
        att3_1 = selfattention(conv3_1.shape[3], 3, conv3_1.shape[3]).to(device)
        conv3_1 = att3_1(conv3_1[0]).unsqueeze(0)

        conv3_2_add = self.bn2(self.conv3_2(conv3_1))
        conv3_2 = self.act(conv3_2_add)
        conv_next = self.bn2(self.conv_sp(conv_next))
        conv_next = self.act(conv_next + conv3_2_add)

        conv3_3 = self.act(self.bn2(self.conv3_3(conv_next)))
        conv3_4_add = self.bn2(self.conv3_4(conv3_3))
        conv3_4 = self.act(conv3_4_add)
        conv_next = self.act(conv_next + conv3_4_add)
        #
        conv3_5 = self.act(self.bn2(self.conv3_5(conv_next)))
        conv3_6_add = self.bn2(self.conv3_6(conv3_5))
        conv3_6 = self.act(conv3_6_add)
        conv_next = self.act(conv_next + conv3_6_add)

        conv3_7 = self.act(self.bn2(self.conv3_7(conv_next)))
        conv3_8_add = self.bn2(self.conv3_8(conv3_7))
        conv3_8 = self.act(conv3_8_add)
        conv_next = self.act(conv_next + conv3_8_add)

        conv4_1 = self.act(self.bn3(torch.cat([self.conv4_1x1(conv_next), self.conv4_3x3(conv_next), self.conv4_5x5(conv_next),
                             self.conv4de_con(self.conv4de(conv_next))], 1)))
        #conv4_1 = self.act(self.bn3(self.conv4_1(conv_next)))
        #
        att4_1 = selfattention(conv4_1.shape[3], 3, conv4_1.shape[3]).to(device)
        conv4_1 = att4_1(conv4_1[0]).unsqueeze(0)

        conv4_2_add = self.bn3(self.conv4_2(conv4_1))
        conv4_2 = self.act(conv4_2_add)
        conv_next = self.bn3(self.conv_sp2(conv_next))
        conv_next = self.act(conv_next + conv4_2_add)

        conv4_3 = self.act(self.bn3(self.conv4_3(conv_next)))
        conv4_4_add = self.bn3(self.conv4_4(conv4_3))
        conv4_4 = self.act(conv4_4_add)
        conv_next = self.act(conv_next + conv4_4_add)
        #
        conv4_5 = self.act(self.bn3(self.conv4_5(conv_next)))
        conv4_6_add = self.bn3(self.conv4_6(conv4_5))
        conv4_6 = self.act(conv4_6_add)
        conv_next = self.act(conv_next + conv4_6_add)

        conv4_7 = self.act(self.bn3(self.conv4_7(conv_next)))
        conv4_8_add = self.bn3(self.conv4_8(conv4_7))
        conv4_8 = self.act(conv4_8_add)
        conv_next = self.act(conv_next + conv4_8_add)

        conv4_9 = self.act(self.bn3(self.conv4_9(conv_next)))
        conv4_10_add = self.bn3(self.conv4_10(conv4_9))
        conv4_10 = self.act(conv4_10_add)
        conv_next = self.act(conv_next + conv4_10_add)

        conv4_11 = self.act(self.bn3(self.conv4_11(conv_next)))
        conv4_12_add = self.bn3(self.conv4_12(conv4_11))
        conv4_12 = self.act(conv4_12_add)
        conv_next = self.act(conv_next + conv4_12_add)

        conv5_1 = self.act(self.bn4(torch.cat([self.conv5_1x1(conv_next), self.conv5_3x3(conv_next), self.conv5_5x5(conv_next),
                             self.conv5de_con(self.conv5de(conv_next))], 1)))
        #conv5_1 = self.act(self.bn4(self.conv5_1(conv_next)))
        #
        att5_1 = selfattention(conv5_1.shape[3], 3, conv5_1.shape[3]).to(device)
        conv5_1 = att5_1(conv5_1[0]).unsqueeze(0)

        conv5_2_add = self.bn4(self.conv5_2(conv5_1))
        conv5_2 = self.act(conv5_2_add)
        conv_next = self.bn4(self.conv_sp3(conv_next))
        conv_next = self.act(conv_next + conv5_2_add)
        #
        conv5_3 = self.act(self.bn4(self.conv5_3(conv_next)))
        conv5_4_add = self.bn4(self.conv5_4(conv5_3))
        conv5_4 = self.act(conv5_4_add)
        conv_next = self.act(conv_next + conv5_4_add)
        #
        conv5_5 = self.act(self.bn4(self.conv5_5(conv_next)))
        conv5_6_add = self.bn4(self.conv5_6(conv5_5))
        conv5_6 = self.act(conv5_6_add)


        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)

        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv2_3_down = self.conv2_3_down(conv2_3)
        conv2_4_down = self.conv2_4_down(conv2_4)
        conv2_5_down = self.conv2_5_down(conv2_5)
        conv2_6_down = self.conv2_6_down(conv2_6)

        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv3_4_down = self.conv3_4_down(conv3_4)
        conv3_5_down = self.conv3_5_down(conv3_5)
        conv3_6_down = self.conv3_6_down(conv3_6)
        conv3_7_down = self.conv3_7_down(conv3_7)
        conv3_8_down = self.conv3_8_down(conv3_8)

        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv4_4_down = self.conv4_4_down(conv4_4)
        conv4_5_down = self.conv4_5_down(conv4_5)
        conv4_6_down = self.conv4_6_down(conv4_6)
        conv4_7_down = self.conv4_7_down(conv4_7)
        conv4_8_down = self.conv4_8_down(conv4_8)
        conv4_9_down = self.conv4_9_down(conv4_9)
        conv4_10_down = self.conv4_10_down(conv4_10)
        conv4_11_down = self.conv4_11_down(conv4_11)
        conv4_12_down = self.conv4_12_down(conv4_12)
        #
        #
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)
        conv5_4_down = self.conv5_4_down(conv5_4)
        conv5_5_down = self.conv5_5_down(conv5_5)
        conv5_6_down = self.conv5_6_down(conv5_6)

        #down是第一次卷积1x1成16的
        out1 = self.score_dsn1(conv1_1_down + conv1_2_down)
        out2 = self.score_dsn2(conv2_1_down + conv2_2_down + conv2_3_down + conv2_4_down + conv2_5_down + conv2_6_down)
        out3 = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down + conv3_4_down + conv3_5_down + conv3_6_down + conv3_7_down + conv3_8_down)
        out4 = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down + conv4_4_down + conv4_5_down + conv4_6_down + conv4_7_down + conv4_8_down +
                               conv4_9_down + conv4_10_down + conv4_11_down + conv4_12_down)
        out5 = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down + conv5_4_down + conv5_5_down + conv5_6_down)

        #dsn进行二次卷积，但是卷积的求和

        out2 = F.conv_transpose2d(out2, self.weight_deconv2, stride=2)
        out3 = F.conv_transpose2d(out3, self.weight_deconv3, stride=4)
        out4 = F.conv_transpose2d(out4, self.weight_deconv4, stride=8)
        out5 = F.conv_transpose2d(out5, self.weight_deconv5, stride=8)
        #反卷积，0填充获得更高清的照片，makelinear是插值权重矩阵
        out2 = self._crop(out2, img_h, img_w, 1, 1)
        out3 = self._crop(out3, img_h, img_w, 2, 2)
        out4 = self._crop(out4, img_h, img_w, 4, 4)
        out5 = self._crop(out5, img_h, img_w, 0, 0)
        #数据的统一
        fuse = torch.cat((out1, out2, out3, out4, out5), dim=1)
        fuse = self.score_fuse(fuse)
        results = [out1, out2, out3, out4, out5, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results
