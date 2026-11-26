import torch.nn as nn
from losses import FocalDiceHybridLoss

class CombinedDeepSupLoss(nn.Module):


    def __init__(self, stage1_kwargs, stage2_kwargs, stage1_weight=0.5, stage2_weight=1.0, ds_weights=[1.0, 0.5, 0.25, 0.125]):
        super(CombinedDeepSupLoss, self).__init__()
        self.criterion1 = FocalDiceHybridLoss(**stage1_kwargs)
        self.criterion2 = FocalDiceHybridLoss(**stage2_kwargs)
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight
        self.ds_weights = ds_weights

    def forward(self, out1, out2, ds1_list, ds2_list, vessel_masks, labels):

        loss1 = self.ds_weights[0] * self.criterion1(out1, vessel_masks)


        for i, ds1 in enumerate(ds1_list or []):
            if i + 1 < len(self.ds_weights):
                loss1 += self.ds_weights[i + 1] * self.criterion1(ds1, vessel_masks)


        vessel_mask_only = vessel_masks[:, 1:2, :, :] * (labels.sum(dim=1, keepdim=True) > 0).to(labels.dtype)
        masked_out2 = out2 * vessel_mask_only
        masked_labels = labels * vessel_mask_only
        loss2 = self.ds_weights[0] * self.criterion2(masked_out2, masked_labels, mask=vessel_mask_only)


        for i, ds2 in enumerate(ds2_list or []):
            if i + 1 < len(self.ds_weights):
                masked_ds2 = ds2 * vessel_mask_only
                loss2 += self.ds_weights[i + 1] * self.criterion2(masked_ds2, masked_labels, mask=vessel_mask_only)


        total_loss = self.stage1_weight * loss1 + self.stage2_weight * loss2

        return total_loss, loss1, loss2
