#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

from mod.instrument import Instrument
from mod.dataloader import DataLoader
from mod.config import Config
from mod.net import Trainer
from mod.fuzzer import Fuzzer

def main(argv):
    config = Config(argv[1:])
    Instrument(config)
    loader = DataLoader(config)
    fuzzer = Fuzzer(config)
    fuzzer.serve()

    extra = loader.incremental_load("fuzzer01")
    fuzzer.sync_all(extra)

    dataset = loader.create_dataset()
    trainer = Trainer(dataset)
    trainer.train()

    for idx, (x, y) in enumerate(dataset):
        topk = trainer.topk(x, y)
        print(topk)
    os.wait()

if __name__ == "__main__":
    main(sys.argv)
