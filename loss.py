class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs, targets = probs.view(-1), targets.view(-1)
        intersection = (probs * targets).sum()
        return 1. - (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

def FocalLoss(logits, targets, alpha=0.8, gamma=2):
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt)**gamma * ce_loss).mean()
