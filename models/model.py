import torch
import torch.nn as nn




class Encoder(nn.Module):
    def __init__(self, nc, ngf, ncondf=12, nhd=16):
        super(Encoder, self).__init__()
        self.ncondf = ncondf

        self.in_cond = nn.ConvTranspose2d(10, 2*ncondf, 6, stride=1, padding=0)
        self.cond_bn1 = nn.BatchNorm2d(2*ncondf)
        self.out_cond = nn.ConvTranspose2d(2*ncondf, ncondf, 4, stride=2, padding=0)

        self.in_conv = nn.Conv2d(nc, 4*ngf, 3, stride=2, padding=1, bias=False)  # b, ngf, 14, 14
        self.in_bn = nn.BatchNorm2d(4*ngf + ncondf)

        self.mid_conv1 = nn.Conv2d(4*ngf + ncondf, 2*ngf, 3, stride=2, padding=1, bias=False)  # b, 36, 7, 7
        self.mid_bn1 = nn.BatchNorm2d(2*ngf)

        self.mid_conv2 = nn.Conv2d(2*ngf, ngf, 3, stride=2, padding=1, bias=False) # b, 8, 4, 4
        self.mid_bn2 = nn.BatchNorm2d(ngf)

        self.out_conv = nn.Conv2d(ngf, nhd, 4, stride=1, padding=0) # b, nhd, 1, 1

        self.activation = nn.ReLU()

    def forward(self, x, cond):
        cond = self.activation(self.cond_bn1(self.in_cond(cond.view(-1, cond.size(1), 1, 1))))
        cond = self.out_cond(cond)

        x = self.in_conv(x)
        x = self.activation(self.in_bn(torch.cat((x, cond), dim=1)))
        x = self.activation(self.mid_bn1(self.mid_conv1(x)))
        x = self.activation(self.mid_bn2(self.mid_conv2(x)))
        return self.out_conv(x)






class Decoder(nn.Module):
    def __init__(self, nc, ngf, ncondf=12, nhd=16):
        super(Decoder, self).__init__()

        self.in_cond = nn.ConvTranspose2d(10, 2*ncondf, 6, stride=1, padding=0) # b, 2*ncondf, 6, 6
        self.cond_bn1 = nn.BatchNorm2d(2*ncondf)
        self.out_cond = nn.Conv2d(2*ncondf, ncondf, 3, stride=1, padding=1) #b, ncondf, 6, 6

        self.in_conv1 = nn.ConvTranspose2d(nhd, 2*ngf, 6, stride=1, padding=0, bias=False)
        # b, 2*ngf, 6, 6
        self.in_bn1 = nn.BatchNorm2d(2*ngf)

        self.in_conv2 = nn.Conv2d(2*ngf, 4*ngf, 3, stride=1, padding=1, bias=False)
        # b, 4*ngf, 6, 6
        self.in_bn2 = nn.BatchNorm2d(4*ngf + ncondf)

        self.mid_conv1 = nn.ConvTranspose2d(4*ngf + ncondf, 2*ngf, 5, stride=2, padding=0, bias=False)
        # b, 36, 15, 15
        self.mid_bn1 = nn.BatchNorm2d(2*ngf)

        self.mid_conv2 = nn.ConvTranspose2d(2*ngf, ngf, 3, stride=2, padding=1, bias=False)
        # b, 8, 29, 29
        self.mid_bn2 = nn.BatchNorm2d(ngf)

        self.out_conv = nn.Conv2d(ngf, nc, 4, stride=1, padding=1)
        # b, 1, 28, 28

        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond):

        cond = self.activation(self.cond_bn1(self.in_cond(cond.view(-1, cond.size(1), 1, 1))))
        cond = self.out_cond(cond)

        x = self.activation(self.in_bn1(self.in_conv1(x)))
        x = self.in_conv2(x)
        x = self.activation(self.in_bn2(torch.cat((x, cond), dim=1)))

        x = self.activation(self.mid_bn1(self.mid_conv1(x)))
        x = self.activation(self.mid_bn2(self.mid_conv2(x)))
        return self.sigmoid(self.out_conv(x))



class AutoEncoder(nn.Module):
    def __init__(self, nc, ngf, ncondf=12, nhd=16):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder(nc, ngf, ncondf, nhd)
        self.dec = Decoder(nc, ngf, ncondf, nhd)

    def forward(self, x, enc_cond, dec_cond):
        out = self.enc(x, enc_cond)
        out = self.dec(out, dec_cond)
        return out





class Generator(nn.Module):
    def __init__(self, nc, ngf, nz, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)



class Discriminator(nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf

        self.in_cond = nn.ConvTranspose2d(10, ndf, 4, 1, bias=False)

        # input is (nc) x 28 x 28
        self.in_conv = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)

        # state size. (ndf) x 14 x 14
        self.mid_conv1 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.mid_bn1 = nn.BatchNorm2d(ndf * 2)

        # state size. (ndf*2) x 7 x 7
        self.mid_conv2 = nn.Conv2d(ndf * 2, ndf * 3, 3, 2, 1, bias=False)
        self.mid_bn2 = nn.BatchNorm2d(ndf * 4)

        # state size. (ndf*4) x 4 x 4
        self.mid_conv3 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False)
        self.mid_bn3 = nn.BatchNorm2d(ndf * 8)

        # state size. (ndf*4) x 4 x 4
        self.mid_conv4 = nn.Conv2d(ndf * 8, ndf * 2, 3, 1, 1, bias=False)
        self.mid_bn4 = nn.BatchNorm2d(ndf * 2)

        # state size. (ndf*8) x 4 x 4
        self.out_conv = nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False)

        self.activation = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, input, cond):
        cond = self.in_cond(cond.view(-1, cond.size(1), 1, 1))

        x = self.in_conv(input)

        x = self.activation(self.mid_bn1(self.mid_conv1(x)))
        x = self.mid_conv2(x)
        x = self.activation(self.mid_bn2(torch.cat((x, cond), dim=1)))
        x = self.activation(self.mid_bn3(self.mid_conv3(x)))
        x = self.activation(self.mid_bn4(self.mid_conv4(x)))

        x = self.out_conv(x)

        return self.sigmoid(x)



if __name__ =="__main__":
    import torch
    x = torch.randn(1,8,1,1)
    img = torch.randn(1,1,28,28)
    a = torch.squeeze(x)
    cond = torch.randn(1,10)
    layer = Discriminator(1, 16, 1)
    res = layer(img, x, cond)
    print(res.shape)

