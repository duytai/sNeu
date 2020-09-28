#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import glob
import struct

pwd = os.path.dirname(os.path.realpath(__file__))
target_bin = os.path.join(pwd, "example/target/debug/example")
target_queue = os.path.join(pwd, "example/out/queue")

if __name__ == "__main__":
    stats = dict()
    testcases = glob.glob("%s/*" % target_queue)
    for testcase in testcases[0:1]:
        process = subprocess.Popen(
            [target_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ
        )
        data = open(testcase, "rb").read()
        process.communicate(data)
        cov_file = ".logs/%d.cov" % process.pid
        cov = open(cov_file, "rb").read()
        os.remove(cov_file)

        for i in range(0, len(cov), 21):

            type_size = struct.unpack("<B", cov[i : i + 1])[0]
            branch_id = struct.unpack("<I", cov[i + 1 : i + 5])[0]
            left_value = struct.unpack("<Q", cov[i + 5 : i + 13])[0]
            right_value = struct.unpack("<Q", cov[i + 13 : i + 21])[0]

            distance = left_value - right_value
            if branch_id not in stats:
                stats[branch_id] = [
                    type_size,
                    abs(distance),
                    1 if distance < 0 else 0,
                    1 if distance == 0 else 0,
                    1 if distance > 0 else 0
                ]
            else:
                stats[branch_id][0] = type_size
                stats[branch_id][1] = min(stats[branch_id][1], abs(distance))
                stats[branch_id][2] = max(stats[branch_id][2], 1 if distance < 0 else 0)
                stats[branch_id][3] = max(stats[branch_id][3], 1 if distance == 0 else 0)
                stats[branch_id][4] = max(stats[branch_id][4], 1 if distance > 0 else 0)

    for branch_id, stat in stats.items():
        is_covered = sum(stat[2:]) >= 2
        print(branch_id)
        print(stat)
        print(is_covered)
        print("------")
