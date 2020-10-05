#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from mod.instrument import Instrument
from mod.dataloader import DataLoader
from mod.config import Config
from mod.net import Trainer

def main(argv):
    config = Config(argv[1:])
    Instrument(config)
    loader = DataLoader(config)
    loader.inc("fuzzer01")
    dataset = loader.create_dataset()
    trainer = Trainer(dataset)
    trainer.train()
    for x, y in dataset:
        top_k = trainer.topk(x, y)

if __name__ == "__main__":
    main(sys.argv)
