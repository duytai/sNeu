#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, signal
import sys
import re
import subprocess
import glob
import struct
import random
import torch
import time
import socket
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

pwd = os.path.dirname(os.path.realpath(__file__))
rustc = os.path.join(pwd, "rustc.py") 
fuzzer = os.path.join(pwd, "fuzzer")

in_dir = ""
out_dir = ""
bin_dir = ""
proj = ""

D_IN = 100
D_OUT = 10

target_afl = "target_afl"
target_sneu = "target_sneu"

master = "fuzzer01"
slave = "fuzzer02"

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
        if opt == "-i":
            in_dir = val
        if opt == "-o":
            out_dir = val
        if opt == "-p":
            proj = val
        if opt == "-b":
            bin_dir = val

    assert in_dir, "[x] require -i"
    assert out_dir, "[x] require -o"
    assert proj, "[x] require -p"
    assert bin_dir, "[x] require -b"

    # Is proj a rust project
    toml = os.path.join(proj, "Cargo.toml")
    if not os.path.exists(toml):
        print("[x] %s is not found" % toml)
        sys.exit()

    ## Detect package name
    toml_str = open(toml, "r").read()
    matches = re.findall("name\s*=\s*\"([^\"]+)\"", toml_str)
    if not len(matches):
        print("[x] Package name is not found")
        sys.exit()

    package = matches[0]
    print("[+] Rust package: '%s'" % package)

    if not os.path.exists(os.path.join(bin_dir, target_afl)):
        ## Build fuzz-target for afl 
        print("[+] Build fuzz-target for AFL")
        os.environ["RUSTC"] = rustc
        os.system("cd %s && cargo clean && cargo build" % proj)

        ## Copy bin target to bin folder
        print("[+] Copy %s to %s" % (target_afl, bin_dir))
        os.system("cp %s/target/debug/%s %s/%s" % (proj, package, bin_dir, target_afl))
    else:
        print("[+] Found %s" % target_afl)

    if not os.path.exists(os.path.join(bin_dir, target_sneu)):
        ## Build fuzz-target for afl 
        print("[+] Build fuzz-target for SNEU")
        os.environ["RUSTC"] = rustc
        os.environ["USE_SNEU"] = "1"
        os.system("cd %s && cargo clean && cargo build" % proj)

        ## Copy bin target to bin folder
        print("[+] Copy %s to %s" % (target_sneu, bin_dir))
        os.system("cp %s/target/debug/%s %s/%s" % (proj, package, bin_dir, target_sneu))
    else:
        print("[+] Found %s" % target_sneu)

    ## Run AFL in subprocess

    if os.fork() > 0:
        try:
            logger = open("log.txt", "w")
            process = subprocess.Popen(
                ["afl-fuzz", "-i", in_dir, "-o", out_dir, "-M", master, "%s/%s" % (bin_dir, target_afl)],
                stdin=subprocess.PIPE,
                stdout=logger,
                stderr=logger,
                env=os.environ
            )
            print("[+] Start afl-fuzz %s" % master)
            process.wait()
        except KeyboardInterrupt:
            pass

    else:
        ## Mutate on child process 
        if os.fork() > 0:
            try:
                process = subprocess.Popen("%s %s/target_afl" % (fuzzer, bin_dir), shell=True)
                process.wait()
            except KeyboardInterrupt:
                pass
        else:
            ## Set up queue for fuzzer02
            if not os.path.exists("%s/%s/queue" % (out_dir, slave)):
                os.makedirs("%s/%s/queue" % (out_dir, slave))
            ## Wait for fuzzer is up
            time.sleep(2)
            ## Open socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", 1234))

            ## Create NN
            nn = Net(D_IN, D_OUT).float()
            learning_rate = 1e-4
            loss_fn = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate)

            n_testcases = 0
            n_crashes = 0

            ## Now training + mutating
            random.seed(1)
            stats = dict()

            full_dataset = []

            while True:
                ## Send new testcases and crashes to fuzzer
                testcases = sorted(list(glob.glob("%s/%s/queue/id:*" % (out_dir, master))))
                crashes = sorted(list(glob.glob("%s/%s/queue/crashes" % (out_dir, master))))

                new_testcases = testcases[n_testcases:]
                new_crashes = crashes[n_crashes:]

                for fname in new_testcases + new_crashes:
                    data = open(fname, "rb").read()
                    assert len(data) <= 1024
                    sock.sendall(data)
                    sock.recv(16)
                print("[+] Send %d testcases to fuzzer" % (len(new_testcases) + len(new_crashes)))

                raw_dataset = []

                for testcase in new_testcases + new_crashes:
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
                print("[+] Covered: %d/%d - Target: %d" % (len(c_branches), len(c_branches) + len(u_branches), len(target_branches)))

                for data, sub_stat in raw_dataset:
                    # Create x
                    x = np.zeros(D_IN)
                    assert len(data) <= D_IN
                    for idx, v in enumerate(bytearray(data)):
                        x[idx] = v
                    x = (x - 128.0) / 128.0
                    # Create y
                    y = np.zeros(D_OUT)
                    for branch_id in sub_stat:
                        if branch_id in target_branches:
                            idx = target_branches.index(branch_id)
                            y[idx] = sub_stat[branch_id][1]
                    y = np.clip(y / 255.0, 0, 1.0)
                    # Append
                    full_dataset.append((torch.tensor(x).float(), torch.tensor(y).float()))

                ## TODO
                ## Reset parameters 
                for layer in nn.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

                ## Train on full_dataset 
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
                    if accuracy >= 80:
                        print("[+] Epoch %d: loss: %f - acc: %f" % (epoch, loss.item(), accuracy))
                        break

                ## Mutate on full_dataset
                g_interests = []
                g_crashes = []
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
                        str_len = np.where(data != 0)[0][-1] + 1
                        sock.sendall(bytearray(data[:str_len].tolist()))
                        code, hbn = [int(x) for x in sock.recv(16).strip().decode("utf-8").split(":")[:2]]
                        if hbn > 0:
                            if not code:
                                g_interests.append(bytearray(data[:str_len].tolist()))
                            else:
                                g_crashes.append(bytearray(data[:str_len].tolist()))

                print("[+] GENERATED %d interest - %d crashes" % (len(g_interests), len(g_crashes)))

                ## Write to in_dir and resume AFL 
                if len(g_interests) + len(g_crashes) > 0:
                    pass

                ## Update number of testcases
                n_testcases = len(testcases)
                n_crashes = len(crashes)
                time.sleep(5)
