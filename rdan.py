import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *
import torch.nn.functional as F
from dbpns import Net as DBPNS

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, n_resblock, nFrames, scale_factor):
        super(Net, self).__init__()
        #base_filter=256
        #feat=64
        self.nFrames = nFrames
        
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        
        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='relu', norm=None) # map to L_t 的特殊 feature extraction block
        self.feat1 = ConvBlock(8, base_filter, 3, 1, 1, activation='relu', norm=None) # map cat 在一起的通用的 feature extraction block，所以通道为8个

        ###DBPNS
        self.DBPN = DBPNS(base_filter, feat, num_stages, scale_factor)
                
        #Res-Block1
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None) \
            for _ in range(n_resblock)]
        modules_body1.append(DeconvBlock(base_filter, feat, kernel, stride, padding, activation='relu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)
        
        #Res-Block2
        modules_body2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None) \
            for _ in range(n_resblock)]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='relu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)
        
        #Res-Block3 就是 Net_{D}
        modules_body3 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None) \
            for _ in range(n_resblock)]
        # 接下来的卷积层是downsample的过程，但是我不想要这个层在我的 attention mechanism 中
        modules_body3.append(ConvBlock(feat, base_filter, kernel, stride, padding, activation='relu', norm=None))
        self.res_feat3 = nn.Sequential(*modules_body3)

        # 接下来的内容与上面本来 modules_body3 后面 append 的内容一致，以备不时之需
        # modules_body4 = [ConvBlock(feat, base_filter, kernel, stride, padding, activation='relu', norm=None)]
        # self.res_feat4 = nn.Sequential(*modules_body4)

        modules_body5 = [
            AttentionBlock(base_filter, feat, 16) # feat = 64, base_filter = 256
        ]
        self.am = nn.Sequential(*modules_body5)
        
        #Reconstruction
        self.output = ConvBlock((nFrames-1)*feat, num_channels, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x, neigbor, flow):
        ### initial feature extraction
        feat_input = self.feat0(x)
        feat_frame=[]
        for j in range(len(neigbor)):
            feat_frame.append(self.feat1(torch.cat((x, neigbor[j], flow[j]),1)))

        # 以上的 initial feature extraction 的代码相当于把论文中 figure 2 的结构的外围的 Conv 块都搞定了
        # 以下的 projection 的代码相当于 figure 2 中的 projection module 的部分，也就是 figure 3 中的内容

        ####Projection
        Ht = []
        for j in range(len(neigbor)):
            h0 = self.DBPN(feat_input)
            h1 = self.res_feat1(feat_frame[j]) # res_feat1 指的是 Net_{misr}
            
            e = h0-h1
            e = self.res_feat2(e) # res_feat2 指的是 Net_{res}
            h = h0+e
            Ht.append(h) # Ht 中的结果要在 Reconstruction 的部分中所用

            # 插入 attention module
            d_r = self.res_feat3(h) # decoder's resnet
            am_o = self.am(d_r) # attention module's output
            d_r += am_o
            feature_input = d_r
            # feature_input = self.res_feat4(d_r) # 还原之前的 downsampling 的过程，以便于后面用来 concat

            # feat_input = self.res_feat3(h) # res_feat3 指的是 Net_{D}
        
        ####Reconstruction
        out = torch.cat(Ht,1)
        output = self.output(out) # output 也是一个与卷积有关的函数
        
        return output
