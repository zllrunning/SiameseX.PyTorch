import torch.nn as nn
import torch.nn.functional as F


class Corr_Up(nn.Module):
    """
    SiamFC head
    """
    def __init__(self):
        super(Corr_Up, self).__init__()

    def _conv2d_group(self, x, kernel):
        batch = x.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px = x.view(1, -1, x.size()[2], x.size()[3])
        po = F.conv2d(px, pk, groups=batch)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po

    def forward(self, z_f, x_f):
        if not self.training:
            return 0.1 * F.conv2d(x_f, z_f)
        else:
            return 0.1 * self._conv2d_group(x_f, z_f)