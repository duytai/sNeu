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
    fuzzer = Fuzzer(config, "fuzzer02")
    fuzzer.wake_up()

    while True:
        batch = loader.incremental_load("fuzzer01")
        fuzzer.sync_all(batch)
        dataset, profile = loader.create_dataset()

        if len(dataset):
            trainer = Trainer(dataset)
            trainer.train()

            for idx, (x, y, data) in enumerate(dataset):
                top_k = trainer.top_k(x, y)[:1]
                fuzzer.mutate(data, top_k, profile)

if __name__ == "__main__":
    main(sys.argv)
