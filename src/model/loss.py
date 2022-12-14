from pdb import set_trace as bp
import torch.nn.functional as F
import torch
import torch.nn as nn



def weighted_bce_loss(output, target, w_p=99, w_n=1, epsilon=1e-7):
    loss_pos = -1 * torch.mean(w_p * target * torch.log(output + epsilon))
    loss_neg = -1 * torch.mean(w_n * (1-target) * torch.log((1-output) + epsilon))
    loss = loss_pos + loss_neg
    return loss


def cost_sensetive_bce_loss(output, target, epsilon=1e-7, w_tp=99, w_tn=0, w_fp=1, w_fn=99):

    fn = w_fn * torch.mean(target * torch.log(output+epsilon))
    tp = w_tp * torch.mean(target * torch.log((1-output)+epsilon))
    fp = w_fp * torch.mean((1-target) * torch.log((1-output)+epsilon))
    tn = w_tn * torch.mean((1-target) * torch.log(output+epsilon))
    return -(fn+tp+fp+tn)
	
class FocalLoss(nn.Module):
	# Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
	def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
		super(FocalLoss, self).__init__()
		self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = loss_fcn.reduction
		self.loss_fcn.reduction = 'none'  # required to apply FL to each element
 
	def forward(self, pred, true):
		loss = self.loss_fcn(pred, true)
		# p_t = torch.exp(-loss)
		# loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability
 
		# TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
		pred_prob = torch.sigmoid(pred)  # prob from logits
		p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
		alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
		modulating_factor = (1.0 - p_t) ** self.gamma
		loss *= alpha_factor * modulating_factor
 
		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'sum':
			return loss.sum()
		else:  # 'none'
			return loss
			
def focal_cost_sensetive_bce(pred, true):
	focal_loss = FocalLoss(nn.BCEWithLogitsLoss());
	return focal_loss(pred, true) + 0.1 * cost_sensetive_bce_loss(pred, true);