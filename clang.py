#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

from subprocess import Popen, PIPE

# TODO: read from env
f_libcmppass = "/root/compare-coverage/libcmppass.so"
f_cmp = "/root/compare-coverage/cmpcov.o"
f_clang = "/usr/bin/clang"

# read and modify flags 
argv = sys.argv[1:]
for idx, opt in enumerate(argv):
    if opt == "-c":
        # instrument c file
        argv = argv + ["-Xclang", "-load", "-Xclang", f_libcmppass]
        break
    if opt == "-o":
        out_file = argv[idx + 1].strip()
        if not out_file.endswith(".o"):
            # inject handler 
            argv = argv + [f_cmp]
        break

# forward to clang
env = os.environ.copy()
argv = [f_clang] + argv 
p = Popen(argv, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env)
out, err = p.communicate()

# report feedback
if out:
    print(out.decode('utf-8').strip())
if err:
    print(err.decode('utf-8').strip(), file=sys.stderr)
