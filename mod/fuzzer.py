import os
import sys
import time
import socket
import glob
import subprocess
import re
import shutil

class Fuzzer:
    def __init__(self, config):
        self.config = config
        self.last_id = 0
        self.workspace = os.path.join(self.config.out_dir, self.config.slave)
        self.master_queue = os.path.join(self.config.out_dir, self.config.master, "queue")
        self.queue_dir = os.path.join(self.workspace, "queue")
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
        os.makedirs(self.queue_dir)

    def wake_up(self):
        fuzzer = os.path.join(self.config.pwd, "fuzzer")
        target_afl = os.path.join(self.config.bin_dir, self.config.target_afl)
        master_queue = self.master_queue

        if not os.fork():
            os.environ["SKIP_WATCH"] = "1"
            process = subprocess.Popen("%s %s %s" % (fuzzer, target_afl, master_queue), shell=True)
            process.wait()
            sys.exit(0)

        time.sleep(2)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", 1234))

    def mutate(self, ins, outs, profile):
        print(ins)
        print(outs)
        print(profile)

    def send_one(self, data):
        self.sock.sendall(data)
        code, hbn = [int(x) for x in self.sock.recv(16).strip().decode("utf-8").split(":")[:2]]
        return (code, hbn)
