import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        inputs = torch.clamp(inputs, min=1e-6, max=1.0 - 1e-6)


        bce = -targets * torch.log(inputs)


        loss = self.alpha * torch.pow((1 - inputs), self.gamma) * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        self.register_buffer('sobel_x', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))

    def get_boundaries(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        channels = x.shape[1]
        kernel_x = self.sobel_x.to(x).expand(channels, 1, 3, 3)
        kernel_y = self.sobel_y.to(x).expand(channels, 1, 3, 3)
        edges_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
        edges_y = F.conv2d(x, kernel_y, padding=1, groups=channels)
        return torch.sqrt(edges_x.square() + edges_y.square() + 1e-8)

    def forward(self, pred, target):
        pred_boundary = self.get_boundaries(pred.float())
        target_boundary = self.get_boundaries(target.float())
        pred_boundary = pred_boundary / pred_boundary.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        target_boundary = target_boundary / target_boundary.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return F.mse_loss(pred_boundary, target_boundary)


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x, mask=None):
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

        if mask is not None:
            mask_h = mask[:, :, 1:, :] * mask[:, :, :-1, :]
            mask_w = mask[:, :, :, 1:] * mask[:, :, :, :-1]
            diff_h = diff_h * mask_h
            diff_w = diff_w * mask_w

            h_tv = diff_h.sum() / (mask_h.sum() + 1e-6)
            w_tv = diff_w.sum() / (mask_w.sum() + 1e-6)
        else:
            h_tv = diff_h.mean()
            w_tv = diff_w.mean()

        return h_tv + w_tv


class FocalDiceHybridLoss(nn.Module):


    def __init__(self, alpha=1.0, gamma=2.0, dice_weight=0.5, tv_weight=0.0, boundary_weight=0.0, **kwargs):
        super(FocalDiceHybridLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.tv = TotalVariationLoss()
        self.boundary = BoundaryLoss()

        self.dice_weight = dice_weight
        self.tv_weight = tv_weight
        self.boundary_weight = boundary_weight

    def forward(self, inputs, targets, mask=None):
        f_loss = self.focal(inputs, targets)
        d_loss = self.dice(inputs, targets)
        loss = f_loss + self.dice_weight * d_loss


        if self.boundary_weight > 0:
            b_loss = self.boundary(inputs, targets)
            loss = loss + self.boundary_weight * b_loss

        if self.tv_weight > 0:
            loss = loss + self.tv_weight * self.tv(inputs, mask)

        return loss


class CombinedLoss(nn.Module):


    def __init__(self, stage1_kwargs, stage2_kwargs, stage1_weight=0.5, stage2_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.criterion1 = FocalDiceHybridLoss(**stage1_kwargs)
        self.criterion2 = FocalDiceHybridLoss(**stage2_kwargs)
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight

    def forward(self, out1, out2, vessel_masks, labels):

        loss1 = self.criterion1(out1, vessel_masks)


        vessel_mask_only = vessel_masks[:, 1:2, :, :] * (labels.sum(dim=1, keepdim=True) > 0).to(labels.dtype)
        masked_out2 = out2 * vessel_mask_only
        masked_labels = labels * vessel_mask_only
        loss2 = self.criterion2(masked_out2, masked_labels, mask=vessel_mask_only)


        total_loss = self.stage1_weight * loss1 + self.stage2_weight * loss2

        return total_loss, loss1, loss2
