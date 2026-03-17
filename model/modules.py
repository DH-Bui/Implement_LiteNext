from .layers import Layernorm

class UpDownstream(nn.Module):
    def __init__(self, scale, in_c, out_c):
        super().__init__()
        self.scale, self.conv = scale, nn.Conv2d(in_c, out_c, 1)
        self.bn, self.ac = Layernorm(out_c), nn.GELU()
      
    def forward(self, x):
        x = self.ac(self.bn(self.conv(x)))
        return F.interpolate(x, scale_factor=self.scale, mode="bilinear")

class SMmoduleplusplus(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, 7, padding="same", groups=in_channel)
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, 1)
        self.conv_local = nn.Conv2d(in_channel, in_channel, 3, padding="same")
        self.ln1, self.ln2, self.ln3 = Layernorm(in_channel), Layernorm(in_channel), Layernorm(in_channel)
      
    def forward(self, x):
        oriin = x
        x = F.gelu(self.ln1(self.conv_local(x)))
        ori = x
        x = F.gelu(self.ln3(self.conv(x)))
        return F.gelu(self.ln2(self.conv1x1(x + ori + oriin)))
