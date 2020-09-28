#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import glob

pwd = os.path.dirname(os.path.realpath(__file__))
target_bin = os.path.join(pwd, "example/target/debug/example")
target_queue = os.path.join(pwd, "example/out/queue")

if __name__ == "__main__":
    testcases = glob.glob("%s/*" % target_queue)
    for testcase in testcases:
        process = subprocess.Popen(
            [target_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ
        )
        data = open(testcase, "rb").read()
        print(data)
        process.communicate(data)
        cov_file = ".logs/%d.cov" % process.pid
        cov = open(cov_file, "rb").read()
        os.remove(cov_file)
        print(cov)

