import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from models import CLIPWrapper

def main(hparams):
    config_dir = 'models/configs/ViT.yaml' if 'ViT' in hparams.model_name else 'models/configs/RN.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]

    model = CLIPWrapper(hparams.model_name, config)
    del hparams.model_name
    trainer = Trainer.from_argparse_args(args, precision=16)
    # TODO create data processors
    # trainer.fit(model, )


if __name__ == '__main__':
    parser = ArgumentParser()
    Trainer.add_argparse_args(parser)
    parser.add_argument('--model_name', required=True)
    args = parser.parse_args()

    main(args)