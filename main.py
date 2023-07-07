import pandas as pd
from torch.utils.checkpoint import checkpoint
import numpy as np
import json
import re
import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torchvision import transforms as tvt
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import math
import os
import pytorch_lightning as pl
import random
from torchtext.data.metrics import bleu_score

from image2latex.model import Image2LatexModel
from data.datamodule import DataModule
from data.dataset import LatexPredictDataset
from data.text import Text100k

emb_dim = 80
dec_dim = 256
enc_dim = 512
attn_dim = 256
lr = 0.001
max_length = 150
log_idx = 300
max_epochs = 15
batch_size = 4
# steps_per_epoch = round(len(train_set) / batch_size)
# total_steps = steps_per_epoch * max_epochs
num_workers = 2
num_layers = 1
drop_out = 0.2
decode = "beamsearch"
beam_width=5
accumulate_batch = 64

text = Text100k()

predict_set = LatexPredictDataset('./samples')

dm = DataModule(None, None, None, predict_set, text=Text100k())

model = Image2LatexModel(
    lr=lr,
    n_class=text.n_class,
    enc_dim=enc_dim,
    emb_dim=emb_dim,
    dec_dim=dec_dim,
    attn_dim=attn_dim,
    num_layers=num_layers,
    dropout=drop_out,
    sos_id=text.sos_id,
    eos_id=text.eos_id,
    decode_type="beamsearch",
    text=text,
    beam_width=beam_width,
    log_step=1,
    log_text=False,
)


lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

accumulate_grad_batches = accumulate_batch //batch_size
trainer = pl.Trainer(
    callbacks=[lr_monitor],
    accelerator="auto",
    log_every_n_steps=1,
    gradient_clip_val=0,
    accumulate_grad_batches=accumulate_grad_batches,
)

trainer.predict(datamodule=dm, model=model, ckpt_path="./model.ckpt")




