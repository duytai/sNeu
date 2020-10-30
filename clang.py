#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

from subprocess import Popen, PIPE

pwd = os.path.dirname(os.path.realpath(__file__))
f_cmp = os.path.join(pwd, "build/cmpcov.o")

# read and modify flags 
argv = sys.argv[1:]
cwd = os.getcwd()

if "-c" in argv:
    argv = argv + ["-fsanitize-coverage=trace-pc-guard,trace-cmp,no-prune"]
elif "-o" in argv:
    argv = argv + [f_cmp, "-fsanitize=address"]

# absoulte path for .c file
argv = [os.path.join(cwd, x) if x.endswith(".c") else x for x in argv]

# forward to clang
env = os.environ.copy()
argv = ["clang"] + argv 
p = Popen(argv, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env)
out, err = p.communicate()

# report feedback
if out:
    print(out.decode('utf-8').strip())
if err:
    print(err.decode('utf-8').strip(), file=sys.stderr)
