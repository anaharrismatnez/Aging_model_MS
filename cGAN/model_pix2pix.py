# Ana Harris 06/02/2023
# Github: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Model based on pix2pix paper: https://arxiv.org/pdf/1611.07004v1.pdf
# Generator: U-net 128 (encoder-decoder)
# Discriminator: patchGAN https://sahiltinky94.medium.com/understanding-patchgan-9f3c8380c207
# FiLM layer: https://ivadomed.org/_modules/ivadomed/models.html#FiLMedUnet

import torch

def convolution_block(in_channels, nf,kernel_size=4,stride=2,padding=1,first_layer = False, last_layer=False):
    if first_layer:
        return torch.nn.Sequential(torch.nn.Conv3d(in_channels,nf,kernel_size,stride,padding), torch.nn.LeakyReLU(0.2,True))
    elif last_layer:
        return torch.nn.Conv3d(in_channels,nf,kernel_size,stride,padding)
    else:
        return torch.nn.Sequential(torch.nn.Conv3d(in_channels,nf,kernel_size,stride,padding), torch.nn.BatchNorm3d(nf), torch.nn.LeakyReLU(0.2,True))


class AttentionGate(torch.nn.Module):
    """
    Additive Attention Gate
    reference: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, in_channels: int):
        super(AttentionGate, self).__init__()

        self.gating_conv =torch.nn.Conv3d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.gating_norm = torch.nn.InstanceNorm3d(in_channels)

        self.input_conv =torch.nn.Conv3d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.input_norm = torch.nn.InstanceNorm3d(in_channels)

        self.relu = torch.nn.ReLU()
        self.conv =torch.nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1)
        self.norm = torch.nn.InstanceNorm3d(1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs: torch.Tensor, shortcut: torch.Tensor):
        g = self.gating_conv(shortcut)
        g = self.gating_norm(g)

        x = self.input_conv(inputs)
        x = self.input_norm(x)

        alpha = torch.add(g, x)
        alpha = self.relu(alpha)
        alpha = self.conv(alpha)
        alpha = self.norm(alpha)
        attention_mask = self.sigmoid(alpha)
        shortcut = torch.mul(attention_mask, shortcut)

        return shortcut


class FiLM_layer(torch.nn.Module):

    def __init__(self,out_nf):
        super(FiLM_layer, self).__init__()

        self.linear_layer = torch.nn.Linear(1,2)
        self.out_nf = out_nf

    def forward(self,delta,X):
        gamma_beta = self.film_layer(delta)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        gamma = gamma.view(-1, self.out_nf, 1, 1, 1)  # Reshape to match feature map dimensions
        beta = beta.view(-1, self.out_nf, 1, 1, 1)
        outputs = gamma * X + beta

        return outputs

class Generator(torch.nn.Module):
    def __init__(self,in_c=1,out_c=1,nf=64,aux_classes=1): 
        super(Generator,self).__init__()
        """ 
        metadata: auxiliary input (delta)
        in_c: number of channels of the input image
        out_c: number of channels in output image
        nf: number of filters in the last conv layer
        """
        self.aux_layer = torch.nn.Linear(aux_classes,aux_classes) 
        self.model = SkipConnection_block(nf*8,nf*8,in_channels=None,previous_block=None,outermost=False,innermost=True)
        self.model = SkipConnection_block(nf*8,nf*8,in_channels=None,previous_block=self.model,outermost=False,innermost=False) 
        self.model = SkipConnection_block(nf*8,nf*8,in_channels=None,previous_block=self.model,outermost=False,innermost=False) 
        self.model = SkipConnection_block(nf*4,nf*8,in_channels=None,previous_block=self.model,outermost=False,innermost=False)
        self.model = SkipConnection_block(nf*2,nf*4,in_channels=None,previous_block=self.model,outermost=False,innermost=False) 
        self.model = SkipConnection_block(nf,nf*2,in_channels=None,previous_block=self.model,outermost=False,innermost=False)
        self.model = SkipConnection_block(out_c,nf,in_channels=in_c+aux_classes,previous_block=self.model,outermost=True,innermost=False)

    def forward(self,X,aux): 
        aux = self.aux_layer(aux).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        aux = aux.repeat(1, 1, X.shape[2], X.shape[3], X.shape[4])
        X = torch.cat([X,aux], dim=1)                                                                        
        X = self.model(X)
        return X


class SkipConnection_block(torch.nn.Module):
    def __init__(self,out_nf,inn_nf,in_channels,previous_block=False,outermost=False,innermost=False):
        """
        out_nf: outter number of filters
        inn_nf: inner number of filters
        in_channels: input channels 
        """
        super(SkipConnection_block,self).__init__()
        if in_channels == None:
            in_channels = out_nf
        downconv = torch.nn.Conv3d(in_channels, inn_nf, kernel_size=4,stride=2, padding=1)
        downrelu = torch.nn.LeakyReLU(0.2, True)
        downnorm = torch.nn.BatchNorm3d(inn_nf)
        uprelu = torch.nn.ReLU(True)
        upnorm =  torch.nn.BatchNorm3d(out_nf)
        upconv = torch.nn.ConvTranspose3d(inn_nf*2,out_nf,kernel_size=4,stride=2,padding=1)
        self.outermost = outermost

        if outermost:
            upsampling = [uprelu,upconv,torch.nn.Tanh()]
            downsampling = [downconv]

            model = torch.nn.Sequential(*downsampling,previous_block,*upsampling)

        elif innermost:
            upconv = torch.nn.ConvTranspose3d(inn_nf,out_nf,kernel_size=4,stride=2,padding=1)
            downsampling = [downrelu,downconv]

            upsampling = [uprelu,upconv,upnorm]

            model =  torch.nn.Sequential(*downsampling,*upsampling)

        else:
            downsampling = [downrelu,downconv,downnorm]

            upsampling = [uprelu,upconv,upnorm,torch.nn.Dropout(0.5)]

            model = torch.nn.Sequential(*downsampling,previous_block,*upsampling)


        self.model = model


    def forward(self,X): 
        if self.outermost:
            X = self.model(X)
            return X
        else:
            outputs = self.model(X) 
            outputs = torch.cat([outputs,X],1)
            return outputs


class Discriminator(torch.nn.Module):
    def __init__(self,p,patch_size,in_c=2,nf=64): 
        super(Discriminator,self).__init__()
        """ 
        in_c: number of channels in input images
        nf: number of filters in the last conv layer
        """

        if patch_size == 70:
            self.N_layers = 5
        elif patch_size == 34:
            self.N_layers = 4
        else:
            self.N_layers = 3

        conv_layers = torch.nn.ModuleList()
        conv_layers.append(convolution_block(in_c,nf,4,2,p,first_layer=True))
        last_c = 1
        for i in range(1,self.N_layers-2):
            conv_layers.append(convolution_block(nf*i,nf*2*i,4,2,p))

            last_c = 2*i

        conv_layers.append(convolution_block(nf*last_c,nf*2*last_c,4,1,p))
        conv_layers.append(convolution_block(nf*2*last_c,1,4,1,p,last_layer=True))
        self.model = torch.nn.Sequential(*conv_layers)


    def forward(self,X):
        intermediate_fmaps = []
        X = self.model[0](X)
        intermediate_fmaps.append(X)

        for i in range(1,self.N_layers-1):
            X = self.model[i](X)
            intermediate_fmaps.append(X)
        X = self.model[self.N_layers-1](X)
        return [X, intermediate_fmaps]


G = Generator(1,1,64,1)
print(G)