#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, signal
import sys
import re
import subprocess
import glob
import struct
import random
import torch
import time
import socket
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mod.instrument import Instrument
from mod.dataloader import DataLoader
from mod.config import Config

def main(argv):
    config = Config(argv[1:])
    Instrument(config)
    loader = DataLoader(config)
    loader.inc("fuzzer01")
    loader.uncovered_branches()

if __name__ == "__main__":
    main(sys.argv)
