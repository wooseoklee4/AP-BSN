
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_loss


eps = 1e-6

# ============================ #
#      Reconstruction loss     #
# ============================ #

@regist_loss
class L1():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        return F.l1_loss(output, data['clean'])

@regist_loss
class L2():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        return F.mse_loss(output, data['clean'])
