import numpy as np
import pymia.data.assembler as asmbl
import torch
import torch.nn.functional as F

import pc.configuration.config as cfg


def on_sample_fn(params: dict):
    key = params['key']
    batch = params['batch']
    idx = params['batch_idx']

    data = params[key]
    # todo: not really clean to use torch but if TensorFlow is used, ops are added to the graph
    data = F.softmax(torch.Tensor(data), 1).numpy()  # convert to probabilities

    index_expr = batch['index_expr'][idx]
    return data, index_expr


def init_shape(shape, id_, batch, idx):
    # shape = list(shape)  # the shape will be of the size of the image, e.g. (60, 330, 384, no_points)
    return np.zeros((batch['size'][idx], cfg.NO_CLASSES), np.float32)


def init_subject_assembler():
    return asmbl.SubjectAssembler(zero_fn=init_shape, on_sample_fn=on_sample_fn)
