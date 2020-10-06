import os
import sys
import time
import socket
import subprocess

class Fuzzer:
    def __init__(self, config):
        self.config = config

    def serve(self):
        fuzzer = os.path.join(self.config.pwd, "fuzzer")
        target_afl = os.path.join(self.config.bin_dir, self.config.target_afl)

        if not os.fork():
            process = subprocess.Popen("%s %s" % (fuzzer, target_afl), shell=True)
            process.wait()
            sys.exit(0)

        time.sleep(2)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", 1234))

    def sync_one(self, data):
        pass

    def sync_all(self, batch):
        ret = []
        for data in batch:
            self.sock.sendall(data)
            code, hbn = [int(x) for x in self.sock.recv(16).strip().decode("utf-8").split(":")[:2]]
            ret.append((code, hbn))
        return ret
