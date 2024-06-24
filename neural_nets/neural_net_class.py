import torch
import torch.nn as nn

class AE_CNN_maxPool(nn.Module):
    def __init__(self):
        super(AE_CNN_maxPool, self).__init__()
        self.channel_mult = 8
        
        # Encoder convolutions
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=self.channel_mult*1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(5, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 2, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4, stride=1, padding=1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1, 2, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*1, self.channel_mult*1, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*1, 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AE_CNN_maxPool2(nn.Module):
    def __init__(self):
        super(AE_CNN_maxPool2, self).__init__()
        self.channel_mult = 8
        
        # Encoder convolutions
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=self.channel_mult*1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(5, stride=2, padding=1),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 2, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4, stride=1, padding=1),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1, 2, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*1, self.channel_mult*1, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*1, 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class AE_CNN_maxPool3(nn.Module):
    def __init__(self):
        super(AE_CNN_maxPool3, self).__init__()
        self.channel_mult = 8
        
        # Encoder convolutions
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=self.channel_mult*1, kernel_size=(10,11), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(5, stride=2, padding=1),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, (10,11), 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4, stride=1, padding=1),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, (10,11), 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*4, 1, 1, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder linear layer
        self.linear_encoder = nn.Sequential(
            nn.Linear(5950, 2975),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2975, 1000),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder linear layer
        self.linear_decoder = nn.Sequential(
            nn.Linear(1000, 2975),
            nn.ReLU(True),
            nn.Linear(2975, 5950),
            nn.ReLU(True),
        )

        # Decoder convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, self.channel_mult*4, 1, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, (10,11), 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1, (10,11), 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*1, self.channel_mult*1, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*1, 2, (34,32), 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        
        shape=x.shape

        x = x.view(-1)
        x = self.linear_encoder(x)
        print('# of features in bottleneck:', x.shape)
        x = self.linear_decoder(x)
        x = x.view(shape)

        x = self.decoder(x)
        return x




class CL_maxPool(nn.Module):
    def __init__(self, channel_mult, h_kernel, w_kernel):
        super(CL_maxPool, self).__init__()
        self.channel_mult = channel_mult
        self.h_kernel = h_kernel
        self.w_kernel = w_kernel
        
        # Encoder convolutions
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=self.channel_mult*1, kernel_size=(10,11), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(5, stride=2, padding=1),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, (10,11), 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4, stride=1, padding=1),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, (10,11), 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*4, 1, 1, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, self.channel_mult*4, 1, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, (10,11), 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1, (10,11), 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*1, self.channel_mult*1, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*1, 2, (h_kernel,w_kernel), 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class FC_variable(nn.Module):
    def __init__(self, num_fc_nodes_after_conv, num_fc_nodes_bottleneck):
        super(FC_variable, self).__init__()
        self.num_fc_nodes_intermediate_layer = int((num_fc_nodes_after_conv-num_fc_nodes_bottleneck)/2)

        # Encoder linear layer
        self.linear_encoder = nn.Sequential(
            nn.Linear(num_fc_nodes_after_conv, self.num_fc_nodes_intermediate_layer),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.num_fc_nodes_intermediate_layer, num_fc_nodes_bottleneck),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder linear layer
        self.linear_decoder = nn.Sequential(
            nn.Linear(num_fc_nodes_bottleneck, self.num_fc_nodes_intermediate_layer),
            nn.ReLU(True),
            nn.Linear(self.num_fc_nodes_intermediate_layer, num_fc_nodes_after_conv),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.linear_encoder(x)
        # print('# of features in bottleneck:', x.shape)
        x = self.linear_decoder(x)
        return x


class AE_net(nn.Module):
    def __init__(self, channel_mult, h_kernel, w_kernel, num_fc_nodes_after_conv, num_fc_nodes_bottleneck):
        super(AE_net, self).__init__()

        self.enc_conv= CL_maxPool(channel_mult, h_kernel, w_kernel).encoder
        self.dec_conv= CL_maxPool(channel_mult, h_kernel, w_kernel).decoder

        self.linear_bottleneck= FC_variable(num_fc_nodes_after_conv, num_fc_nodes_bottleneck)

    def forward(self, x):

        x = self.enc_conv(x)
        shape = x.shape

        x = x.view(-1)
        x = self.linear_bottleneck(x)
        x = x.view(shape)

        x = self.dec_conv(x)
        return x



