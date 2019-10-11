import random

import numpy as np
import tensorflow as tf
import torch


def set_seed(seed: int, cudnn_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
