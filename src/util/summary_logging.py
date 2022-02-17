
import time

from torch.utils.tensorboard import SummaryWriter
import numpy as np

class LossWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment=''):
        if log_dir == None:
            log_dir = './logs/tensorboard/' + time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime(time.time()))
        super(LossWriter, self).__init__(log_dir=log_dir, comment=comment)

    def write_loss(self, loss_name, scalar, n_iter):
        self.add_scalar('Loss/'+loss_name, scalar, n_iter)


if __name__=='__main__':
    testwriter = LossWriter()

    for n_iter in range(100):
        testwriter.write_loss(np.random.random(), n_iter)
