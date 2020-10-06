import os
import sys
import time
import socket
import glob
import subprocess
import re
import shutil

class Fuzzer:
    def __init__(self, config, name):
        self.config = config
        self.last_id = 0
        self.workspace = os.path.join(self.config.out_dir, name)
        self.queue_dir = os.path.join(self.workspace, "queue")
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
        os.makedirs(self.queue_dir)

    def mutate(self, data, top_k, profile):
        last_id = self.last_id
        for pos in top_k:
            tmp = data + (pos - (len(data) - 1)) * b"\x00"
            for p in profile:
                tmp = tmp[0:pos] + bytes([tmp[pos] ^ p % 256]) + tmp[pos + 1:]
                code, hbn = self.send_one(tmp)
                if not code and hbn > 0:
                    self.last_id += 1
                    f = open("%s/id:%06d,src:sneu" % (self.queue_dir, self.last_id), "wb")
                    f.write(tmp)
                    f.close()
        print("[+] Fuzzer: generated %d testcases" % (self.last_id - last_id))

    def wake_up(self):
        fuzzer = os.path.join(self.config.pwd, "fuzzer")
        target_afl = os.path.join(self.config.bin_dir, self.config.target_afl)

        if not os.fork():
            process = subprocess.Popen("%s %s" % (fuzzer, target_afl), shell=True)
            process.wait()
            sys.exit(0)

        time.sleep(2)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", 1234))

    def send_one(self, data):
        self.sock.sendall(data)
        code, hbn = [int(x) for x in self.sock.recv(16).strip().decode("utf-8").split(":")[:2]]
        return (code, hbn)

    def sync_all(self, batch):
        ret = []
        for data in batch:
            ret.append(self.send_one(data))
        print(ret)
        return ret
