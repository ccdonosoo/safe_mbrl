from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime

class Logger(object):
    
    def __init__(self, log_dir, **kwargs):
        runtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        log_dir = f"{log_dir}/{runtime}"
        
        self.writer = SummaryWriter(log_dir, **kwargs)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images.
        Args: images: numpy of shape (Batch x C x H x W) in the range [-1.0, 1.0]
        """

        if type(images) == tuple or type(images) == list:
            images = np.array(images)

        if len(images.shape) == 3:
            images = images[:,:,:,None]
            self.writer.add_images("{}".format(tag), images, global_step=step, dataformats="NHWC")
            return    
       
        self.writer.add_images("{}".format(tag), images, global_step=step, dataformats="NHWC")
           
    def histo_summary(self, tag, values, step, bins="auto"):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram('{}'.format(tag), values, bins=bins, global_step=step)