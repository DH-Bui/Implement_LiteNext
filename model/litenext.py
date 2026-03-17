import copy
import torch
from .layers import RB, MLP, EMA
from .modules import UpDownstream, SMmoduleplusplus

class Feature_extractorLGMixer(nn.Module):
    ...

class LiteNextNet(nn.Module):
    def __init__(self, in_channel=16):
        super().__init__()
        self.student = Feature_extractorLGMixer(in_channel)
        self.teacher = copy.deepcopy(self.student)
        self.predictor = MLP(in_channel*8, in_channel*8)
        self.ema_updater = EMA(0.99)
        self.bottle = RB(in_channel*8, in_channel*8)
        self.up3 = UpDownstream(2, in_channel*8, in_channel*4)
        self.up2 = UpDownstream(2, in_channel*4, in_channel*2)
        self.up1 = UpDownstream(2, in_channel*2, in_channel)
        self.up0 = UpDownstream(2, in_channel, in_channel)
        self.head = nn.Conv2d(in_channel, 1, 1)

    def forward(self, x_s, x_w=None, phase="train"):
        e1, e2, e3, e4 = self.student(x_s)
        loss_con = torch.tensor(0.0).to(x_s.device)
        if phase == "train" and x_w is not None:
            s_pred = self.predictor(e4)
            with torch.no_grad():
                _, _, _, t_e4 = self.teacher(x_w)
                t_pred = (F.adaptive_avg_pool2d(t_e4, (1,1)) + F.adaptive_max_pool2d(t_e4, (1,1))).view(t_e4.shape[0], -1)
            loss_con = torch.mean(1 - F.cosine_similarity(s_pred, t_pred))

        d = self.up3(self.bottle(e4))
        d = self.up2(d)
        d = self.up1(d)
        d = self.up0(d)
        return self.head(d), loss_con

    def update_moving_average(self):
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data = self.ema_updater.update_average(t.data, s.data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LiteNextNet(in_channel=16).to(device)
print(f"Model initialized on {device}")
summary(model, (3, 256, 256))
