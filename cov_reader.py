#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import struct
import subprocess
import os
import re

test_bin = os.environ["BIN"] if "BIN" in os.environ else os.path.join(os.getcwd(), "test")

if __name__ == '__main__':
    ## extract total number of comparison instructions 
    my_env = os.environ.copy()
    process = subprocess.Popen(
        ["strings", test_bin],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=my_env
    )
    outs, errs = process.communicate()
    branches = []
    if outs:
        outs = outs.decode('utf-8').strip()
        matches = re.findall("__sn_\d+_\d+", outs)
        for match in matches:
            offset, size = [int(x) for x in match.split("_")[-2:]]
            branches = branches + list(range(offset, offset + size))
        branches = sorted(branches)
        print("+ comparisons: %d" % (len(branches)))
    if errs:
        print(errs.decode('utf-8').strip())

    ## try with one test 
    process = subprocess.Popen(
        [test_bin],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=my_env
    )
    process.communicate()

    cov_file = ".logs/%d.cov" % process.pid
    data = open(cov_file, "rb").read()

    label = [0] * len(branches)
    print(branches)
    for i in range(0, len(data), 25):
        type_size = struct.unpack("<B", data[i : i + 1])[0]
        branch_id = struct.unpack("<Q", data[i + 1 : i + 9])[0]
        left_value = struct.unpack("<Q", data[i + 9 : i + 17])[0]
        right_value = struct.unpack("<Q", data[i + 17 : i + 25])[0]
        distance = left_value - right_value
        label[branches.index(branch_id)] += distance
    os.remove(cov_file)

    print(label)
