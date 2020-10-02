#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import os 

if __name__ == "__main__":
    argv = sys.argv[1:]
    in_dir = ""

    for idx in range(0, len(argv) - 1, 2):
        opt, val = argv[idx], argv[idx + 1]
        if opt == "-i":
            in_dir = val
        if opt == "-o":
            out_dir = val
        if opt == "-b":
            bin_dir = val

    assert out_dir, "[x] require -o"
    assert bin_dir, "[x] require -b"

    if in_dir:
        ## Start new afl-fuzz instance 
        process = subprocess.Popen(
            ["afl-fuzz", "-i", in_dir, "-o", out_dir, "%s/target_afl" % bin_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ
        )
    else:
        ## Resume
        process = subprocess.Popen(
            ["afl-fuzz", "-i-", "-o", out_dir, "%s/target_afl" % bin_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ
        )
        
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip().decode("utf-8"))
