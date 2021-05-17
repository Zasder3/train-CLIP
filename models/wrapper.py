import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lighting as pl

class Wrapper(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 model_name: str,
                 isViT: bool
                 ):
        super().__init__()

        self.model = model
        self.model_name = model_name
        self.isViT = isViT
        self.image_loss = nn.CrossEntropyLoss()
        self.text_loss = nn.CrossEntropyLoss()
    
    def forward(self, image, text):
        return self.model(image, text)
    
    # source: https://github.com/openai/CLIP/issues/83
    def training_step(self, train_batch, idx):
        image, text = train_batch
        image_logits, text_logits = self.forward(image, text)
        ground_truth = torch.arange(len(image_logits))
        loss = (self.image_loss(image_logits, ground_truth) + self.text_loss(text_logits, ground_truth)).div(2)
        self.log('train_loss', loss)
    
    def validation_step(self, val_batch, idx):
        image, text = val_batch
        image_logits, text_logits = self.forward(image, text)
        ground_truth = torch.arange(len(image_logits))
        loss = (self.image_loss(image_logits, ground_truth) + self.text_loss(text_logits, ground_truth)).div(2)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        lr = {
            "RN50": 5e-4,
            "RN101": 5e-4,
            "RN50x4": 5e-4,
            "RN50x16": 4e-4,
            "RN50x64": 3.6e-4,
            "ViT-B/32": 5e-4,
            "ViT-B/16": 5e-4,
            "ViT-L/14": 4e-4,
            "ViT-L/14-336px": 2e-5,
        }[self.model_name]

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(
                0.9,
                0.98 if self.isViT else 0.999
            ),
            eps=1e-6 if self.isViT else 1e-8,
            weight_decay=0.2
        )

        # TODO Hear back from authors about this
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2000
        )

        opt_dict = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

        return opt_dict
