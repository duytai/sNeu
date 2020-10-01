#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import glob
import struct
import random
import torch
import sys
import socket
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

D_IN = 100
D_OUT = 10
pwd = os.path.dirname(os.path.realpath(__file__))

class Net(nn.Module):

    def __init__(self, d_in, d_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d_in, d_in)
        self.fc2 = nn.Linear(d_in, 32)
        self.fc3 = nn.Linear(32, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def is_match(y, y_pred):
    temp = (y > 0.5) == (y_pred > 0.5)
    temp = temp.type(torch.float)
    return 1 if temp.sum() == len(y) else 0

def parse_coverage(cov, stats):
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

if __name__ == "__main__":

    argv = sys.argv[1:]

    for idx in range(0, len(argv), 2):
        opt, val = argv[idx], argv[idx + 1]
        if opt == "-o":
            out_dir = val
        if opt == "-b":
            bin_dir = val

    assert out_dir, "[x] require -o"
    assert bin_dir, "[x] require -b"
    ## Open socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 1234))

    ## Send all testcases and crashes to fuzzer
    testcases = sorted(list(glob.glob("%s/queue/id:*" % out_dir)))
    for testcase in testcases:
        data = open(testcase, "rb").read()
        assert len(data) <= 1024


    assert False

    random.seed(1)
    stats = dict()
    raw_dataset = []
    full_dataset = []

    testcases = sorted(list(glob.glob("%s/queue/*" % out_dir)))
    print("[+] num testcases: %d" % len(testcases))
    for testcase in testcases:
        sub_stat = dict()
        process = subprocess.Popen(
            ["%s/target_sneu" % bin_dir],
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
        parse_coverage(cov, stats)
        parse_coverage(cov, sub_stat)
        raw_dataset.append((data, sub_stat))

    c_branches = [branch_id for branch_id in stats if sum(stats[branch_id][2:]) >= 2] 
    u_branches = [branch_id for branch_id in stats if sum(stats[branch_id][2:]) < 2]
    target_branches = random.choices(u_branches, k=D_OUT)

    print("[+] covers: %d" % len(c_branches))
    print("[+] uncovers: %d" % len(u_branches))
    print("[+] target: %d" % len(target_branches))

    # create training raw_dataset 
    for data, sub_stat in raw_dataset:
        x = np.zeros(D_IN)
        # create x
        assert len(data) <= D_IN
        for idx, v in enumerate(bytearray(data)):
            x[idx] = v
        x = (x - 128.0) / 128.0
        # create y
        y = np.zeros(D_OUT)
        for branch_id in sub_stat:
            if branch_id in target_branches:
                idx = target_branches.index(branch_id)
                y[idx] = sub_stat[branch_id][1]
        y = np.clip(y / 255.0, 0, 1.0)
        full_dataset.append((torch.tensor(x).float(), torch.tensor(y).float()))

    # training
    nn = Net(D_IN, D_OUT).float()
    learning_rate = 1e-4
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate)

    ## training
    for epoch in range(100):
        accuracy = 0
        for (x, y) in full_dataset:
            y_pred = nn(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accuracy += is_match(y, y_pred) / len(full_dataset) * 100
        print("[+] epoch %d: loss: %f - acc: %f" % (epoch, loss.item(), accuracy))
        if accuracy >= 80:
            break

    ## mutate 
    idx = len(full_dataset)
    for (x, y) in full_dataset:
        x.requires_grad = True
        y_pred = nn(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            top_k = np.array(x.grad).argsort()[-5:][::-1]
            data = x * 128.0 + 128.0
            data = data.int().numpy()
            for k in top_k:
                data[k] = 1

    #  process = subprocess.Popen(
        #  [fuzzer],
        #  stdin=subprocess.PIPE,
        #  stdout=subprocess.PIPE,
        #  stderr=subprocess.PIPE,
        #  env=os.environ
    #  )
