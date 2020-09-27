#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

from subprocess import Popen, PIPE

pwd = os.path.dirname(os.path.realpath(__file__))

argv = sys.argv[1:]
argv = ["rustc"] + argv + [
    "-C", "passes=sancov",
    "-C", "llvm-args=-sanitizer-coverage-level=3",
    "-C", "llvm-args=-sanitizer-coverage-prune-blocks=0",
    "-C", "opt-level=3",
    "-C", "target-cpu=native",
    "-L", pwd,
    "-l", "cmpcov",
]

with open("./log.txt", "w") as f:
    f.write(str(argv))
    f.close()

p = Popen(argv, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=os.environ)
out, err = p.communicate()

# report feedback
if out:
    print(out.decode('utf-8').strip())
if err:
    print(err.decode('utf-8').strip(), file=sys.stderr)
