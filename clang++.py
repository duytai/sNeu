#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

from subprocess import Popen, PIPE

pwd = os.path.dirname(os.path.realpath(__file__))
cmpcov = os.path.join(pwd, "build/cmpcov.o")

# read and modify flags 
argv = sys.argv[1:] + ["-fsanitize-coverage=trace-pc-guard,trace-cmp", cmpcov] 

# forward to clang
env = os.environ.copy()
argv = ["clang++"] + argv 
p = Popen(argv, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env)
out, err = p.communicate()

# report feedback
if out:
    print(out.decode('utf-8').strip())
if err:
    print(err.decode('utf-8').strip(), file=sys.stderr)
