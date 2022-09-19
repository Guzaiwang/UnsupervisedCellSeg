from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
log_dir = './output_test'

log_writer = SummaryWriter(log_dir=log_dir)


img_path = './demo/0_0.png'
img = Image.open(img_path)
img = np.array(img)/65536. * 255.
print(np.max(img.astype(np.uint8)))
print(np.min(img.astype(np.uint8)))

log_writer.add_images('origin_img', img.astype(np.uint8), dataformats='HW')


log_writer.flush()
