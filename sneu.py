#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time

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
    fuzzer.wake_up()

    batch = loader.incremental_load()
    dataset, profile = loader.create_dataset()

    if len(dataset):
        trainer = Trainer(dataset)
        trainer.train()

        ins = []
        outs = []

        for idx, (x, y, data) in enumerate(dataset):
            top_k = trainer.top_k(x, y)
            ins.append(data)
            outs.append(top_k)

        fuzzer.mutate(ins, outs, profile)

    os.wait()

if __name__ == "__main__":
    main(sys.argv)
