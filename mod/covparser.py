import subprocess
import os
import struct

class CovParser:
    def __init__(self, config):
        self.config = config

    def parse(self, data):
        process = subprocess.Popen(
            ["%s/target_sneu" % self.config.bin_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ
        )
        process.communicate(data)
        cov_file = ".logs/%d.cov" % process.pid
        cov = open(cov_file, "rb").read()
        os.remove(cov_file)
        ret = []
        for i in range(0, len(cov), 21):
            type_size = struct.unpack("<B", cov[i : i + 1])[0]
            branch_id = struct.unpack("<I", cov[i + 1 : i + 5])[0]
            left_value = struct.unpack("<Q", cov[i + 5 : i + 13])[0]
            right_value = struct.unpack("<Q", cov[i + 13 : i + 21])[0]
            ret.append((type_size, branch_id, left_value, right_value))
        return ret
