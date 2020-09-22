#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import glob
import re
import subprocess
import struct
import pickle
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# TODO: use relative path
TARGET_BIN = "/root/afl-setup/binutils-sn/binutils/readelf"
TARGET_QUEUE = "/root/afl-setup/out/queue"
WINDOW_SIZE = 10

class Net(nn.Module):
    def __init__(self, d_in, d_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_out, d_out)
        self.fc3 = nn.Linear(d_out, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def extract_branches():
    my_env = os.environ.copy()
    process = subprocess.Popen(
        ["readelf", "-s", TARGET_BIN],
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
    if errs:
        print(errs.decode('utf-8').strip())
    
    return np.array(branches)

def construct_x(testcase):
    data = open(testcase, "rb").read()
    data = [x for x in bytearray(data)] + (1024 * 20 - len(data)) * [0]
    return (np.array(data) - 128) / 128

def construct_y(testcase, branches):
    # read branch distance
    Y_MAX = 32768

    my_env = os.environ.copy()
    process = subprocess.Popen(
        [TARGET_BIN, "-a", testcase],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=my_env
    )
    process.communicate()
    cov_file = ".logs/%d.cov" % process.pid
    data = open(cov_file, "rb").read()
    os.remove(cov_file)

    y = np.array([Y_MAX] * len(branches))
    for i in range(0, len(data), 19):
        type_size = struct.unpack("<B", data[i : i + 1])[0]
        branch_id = struct.unpack("<H", data[i + 1 : i + 3])[0]
        left_value = struct.unpack("<Q", data[i + 3 : i + 11])[0]
        right_value = struct.unpack("<Q", data[i + 11 : i + 19])[0]
        distance = abs(left_value - right_value)
        idx = np.where(branches == branch_id)[0][0]
        if y[idx] == Y_MAX:
            y[idx] = distance
        else:
            y[idx] += distance
    return np.clip(y / Y_MAX, 0, 1)

def construct_xy(testcase, branches):
    y = construct_y(testcase, branches)
    x = construct_x(testcase)
    return (x, y)

def is_match(y, y_pred):
    temp = (y > 0.5) == (y_pred > 0.5)
    temp = temp.type(torch.float)
    return 1 if temp.sum() == len(y) else 0

if __name__ == "__main__":
    # get total number of branches
    branches = extract_branches()
    print("[+] Number of branches: %d" % len(branches))

    # sample dataset to get dimension
    while True:
        testcases = sorted(glob.glob(os.path.join(TARGET_QUEUE, "*")))
        print("[+] Numer of testcases: %d" % len(testcases))
        if len(testcases) > 0:
            x, y = construct_xy(testcases[0], branches)
            net = Net(x.shape[0], y.shape[0]).float()
        if len(testcases) > WINDOW_SIZE:
            slide_window = [construct_xy(testcase, branches) for testcase in testcases[0:WINDOW_SIZE]]
            break
        else:
            time.sleep(1)

    print("[+] WINDOW_SIZE: %d" % len(slide_window))
    print("[+] Net is ready")
    print(net)

    # read data and train online
    learning_rate = 1e-4
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # online training
    total = 1
    matches = 1
    idx = 0

    while True:
        testcases = sorted(glob.glob(os.path.join(TARGET_QUEUE, "*")))[WINDOW_SIZE:]
        if idx < len(testcases):
            x, y = construct_xy(testcases[idx], branches)
            sampling = random.choices(slide_window, k=4) + [(x, y)]
            slide_window.append((x, y))
            slide_window.pop(0)
            for x, y in sampling:
                y_pred = net(torch.tensor(x).float())
                loss = loss_fn(y_pred, torch.tensor(y).float())
                optimizer.zero_grad()
                loss.backward()
                with torch.no_grad():
                    idx = idx + 1
                    total = total + 1
                    matches = matches + is_match(torch.tensor(y).float(), y_pred) 
                    acc = matches * 100 / total
                    print("%d/%d:Accuracy: %f - Loss: %f" % (idx / 5, len(testcases), acc, loss.item()))
        time.sleep(1)
