import torch
from torch import nn
import pytorch_lightning as pl

from data.text import Text
from image2latex.image2latex import Image2Latex

class Image2LatexModel(pl.LightningModule):
    def __init__(
        self,
        lr,
        n_class: int,
        enc_dim: int = 512,
        emb_dim: int = 80,
        dec_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        decode_type: str = "greedy",
        text: Text = None,
        beam_width: int = 5,
        sos_id: int = 1,
        eos_id: int = 2,
        log_step: int = 100,
        log_text: bool = False,
    ):
        super().__init__()
        self.model = Image2Latex(
            n_class,
            enc_dim,
            emb_dim,
            dec_dim,
            attn_dim,
            num_layers,
            dropout,
            bidirectional,
            decode_type,
            text,
            beam_width,
            sos_id,
            eos_id,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.text = text
        self.max_length = 150
        self.log_step = log_step
        self.log_text = log_text
        self.save_hyperparameters()
        self.pairs = []

    def forward(self, images, formulas, formula_len):
        return self.model(images, formulas, formula_len)

    # Do things u want here at predict step
    def predict_step(self, batch, batch_idx):
        image, name = batch

        name = name[0][:-4]
        latex = self.model.decode(image, self.max_length)

        with open(f'./output/{name}.txt', 'w') as f:
            f.write(latex)

        return latex
