

class Image2LatexModel(pl.LightningModule):
    def __init__(
        self,
        lr,
        total_steps,
        n_class: int,
        enc_dim: int = 512,
        enc_type: str = "conv_row_encoder",
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
            enc_type,
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
        self.total_steps = total_steps
        self.text = text
        self.max_length = 150
        self.log_step = log_step
        self.log_text = log_text
#         self.exact_match = load("exact_match")
        self.save_hyperparameters()
        self.pairs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=self.total_steps, verbose=False,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def forward(self, images, formulas, formula_len):
        return self.model(images, formulas, formula_len)

    def training_step(self, batch, batch_idx):
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        images, formulas, formula_len, image_names = batch

        formulas_in = formulas[:, :-1]
        formulas_out = formulas[:, 1:]

        outputs = self.model(images, formulas_in, formula_len)

        bs, t, _ = outputs.size()
        _o = outputs.reshape(bs * t, -1)
        _t = formulas_out.reshape(-1)
        loss = self.criterion(_o, _t)

        self.log("train loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, formulas, formula_len, _ = batch

        formulas_in = formulas[:, :-1]
        formulas_out = formulas[:, 1:]

        outputs = self.model(images, formulas_in, formula_len)

        bs, t, _ = outputs.size()
        _o = outputs.reshape(bs * t, -1)
        _t = formulas_out.reshape(-1)

        loss = self.criterion(_o, _t)

        predicts = [
            self.text.tokenize(self.model.decode(i.unsqueeze(0), self.max_length))
            for i in images
        ]
        truths = [self.text.tokenize(self.text.int2text(i)) for i in formulas]

        bleu4 = torch.mean(
            torch.Tensor(
                [bleu_score([pre], [[tru]]) for pre, tru in zip(predicts, truths)]
            )
        )


        if self.log_text and batch_idx % self.log_step == 0:
            for truth, pred in zip(truths, predicts):
                print("=" * 20)
                print(f"Truth: [{' '.join(truth)}] | Predict: [{' '.join(pred)}]")
                print("=" * 20)                
            print()

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_bleu4", bleu4, sync_dist=True)
#         self.log("val_exact_match", em, sync_dist=True)

        return bleu4, loss

    def test_step(self, batch, batch_idx):
        images, formulas, formula_len, image_names = batch

        formulas_in = formulas[:, :-1]
        formulas_out = formulas[:, 1:]

        outputs = self.model(images, formulas_in, formula_len)

        bs, t, _ = outputs.size()
        _o = outputs.reshape(bs * t, -1)
        _t = formulas_out.reshape(-1)

        loss = self.criterion(_o, _t)

        predicts = [
            self.text.tokenize(self.model.decode(i.unsqueeze(0), self.max_length))
            for i in images
        ]
        truths = [self.text.tokenize(self.text.int2text(i)) for i in formulas]

        bleu4 = torch.mean(
            torch.Tensor(
                [bleu_score([pre], [[tru]]) for pre, tru in zip(predicts, truths)]
            )
        )

        
        columns = ['Truth', 'Predict']

        if True and batch_idx % self.log_step == 0:
            for truth, pred, name in zip(truths, predicts, image_names):
                with open('truth.txt','a') as t:
                    t.write(f"{' '.join(truth)}\n")
                with open('pred.txt', 'a') as f:
                    f.write(f"{' '.join(pred)}\n")
                with open('name.txt', 'a') as f:
                    f.write(f"{name}\n")
                    
             

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_bleu4", bleu4, sync_dist=True)
        return bleu4, loss
    
    # Do things u want here at predict step
    def predict_step(self, batch, batch_idx):
        image = batch

        latex = self.model.decode(image, self.max_length)

        print("Predicted:", latex)

        return latex
