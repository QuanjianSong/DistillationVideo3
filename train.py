import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from trainer import ScoreDistillationTrainer, ODETrainer
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs", help="Path to the directory to save logs")
    args = parser.parse_args()

    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
        OmegaConf.create(vars(args))
    )

    return config


def main():
    config = parse_args()

    if config.trainer == 'score_distillation':
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == 'ode':
        trainer = ODETrainer(config)
    trainer.train()
        


if __name__ == "__main__":
    main()
