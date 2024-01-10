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
        gamma_beta = self.linear_layer(delta)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gamma = gamma.repeat(1, self.out_nf, X.shape[2], X.shape[3], X.shape[4]) # Reshape to match feature map dimensions
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.repeat(1, self.out_nf, X.shape[2], X.shape[3], X.shape[4])
        outputs = gamma * X + beta

        return outputs

class Generator(torch.nn.Module):
    def __init__(self,in_c=1,out_c=1,nf=64,layers=7): 
        super(Generator,self).__init__()
        """ 
        metadata: auxiliary input (delta)
        in_c: number of channels of the input image
        out_c: number of channels in output image
        nf: number of filters in the last conv layer
        """

        self.in_outtermost = torch.nn.Conv3d(in_c,nf, kernel_size=4,stride=2, padding=1)

        filters = [nf * 8 if 2**i >= 8 else nf * (2**i) for i in range(layers-1)] 

        self.encoder_layers = torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()

        for i in range(layers-2):
            self.encoder_layers.append(Encoder(filters[i],filters[i+1],innermost=False))

        self.encoder_layers.append(Encoder(filters[-1],filters[-1],innermost=True))
        self.decoder_layers.append(Decoder(filters[-1],filters[-1],innermost=True))

        for i in range(layers-2,0,-1):
            if i >= (layers-3):
                self.decoder_layers.append(Decoder(filters[i]*2,filters[i-1],innermost=False,use_dropout=True)) ## DROPOUT IN INTERMIDIATE LAYERS
            else:
                self.decoder_layers.append(Decoder(filters[i]*2,filters[i-1],innermost=False))
            
        self.out_outtermost = torch.nn.Sequential(torch.nn.ReLU(True),
            torch.nn.ConvTranspose3d(nf*2,out_c,kernel_size=4,stride=2,padding=1),
            torch.nn.Tanh()
            )

    def forward(self,X,delta):                                                      
        outputs = self.in_outtermost(X)
        shortcuts = [outputs]
        for i in range(len(self.encoder_layers)):
            outputs = self.encoder_layers[i](outputs,delta)
            shortcuts.append(outputs)

        shortcuts = shortcuts[-2::-1]
        for i in range(len(self.decoder_layers)):
            outputs = self.decoder_layers[i](outputs,shortcuts[i])

        outputs = self.out_outtermost(outputs)

        return outputs


class Encoder(torch.nn.Module):
    def __init__(self,inn_nf,out_nf,innermost=False):
        """
        out_nf: outter number of filters
        inn_nf: inner number of filters
        """
        super(Encoder,self).__init__()

        downconv = torch.nn.Conv3d(inn_nf, out_nf, kernel_size=4,stride=2, padding=1)
        downrelu = torch.nn.LeakyReLU(0.2, True)
        downnorm = torch.nn.BatchNorm3d(out_nf)


        if innermost:
            self.downsampling = torch.nn.Sequential(downrelu,downconv)
            self.film_layer = FiLM_layer(out_nf)

        else:
            self.downsampling = torch.nn.Sequential(downrelu,downconv,downnorm)
            self.film_layer = FiLM_layer(out_nf)

    def forward(self,X,delta):
        X = self.downsampling(X)
        X = self.film_layer(delta,X)
        return X

class Decoder(torch.nn.Module):
    def __init__(self,inn_nf,out_nf,innermost=False,use_dropout=False):
        """
        out_nf: outter number of filters
        inn_nf: inner number of filters
        """
        super(Decoder,self).__init__()

        uprelu = torch.nn.ReLU(True)
        upnorm =  torch.nn.BatchNorm3d(out_nf)
        upconv = torch.nn.ConvTranspose3d(inn_nf,out_nf,kernel_size=4,stride=2,padding=1)

        if innermost:
            self.upsampling =  torch.nn.Sequential(uprelu,upconv,upnorm,torch.nn.Sigmoid())
            #self.attention_gate = AttentionGate(out_nf)

        else:
            if use_dropout:
                self.upsampling = torch.nn.Sequential(uprelu,upconv,upnorm,torch.nn.Dropout(0.5),torch.nn.Sigmoid())
            else:
                self.upsampling = torch.nn.Sequential(uprelu,upconv,upnorm,torch.nn.Sigmoid())

            #self.attention_gate = AttentionGate(out_nf)

    def forward(self,X,shortcut): 
        X = self.upsampling(X)
        #shortcut = self.attention_gate(X,shortcut)
        outputs = torch.cat([X,shortcut],dim=1)

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

""" G = Generator(1,1,64,7)
print(G) """