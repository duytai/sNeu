import os
import glob

from mod.covparser import CovParser

class DataLoader:

    def __init__(self, config):
        self.config = config
        self.dataset = []
        self.max_len = 0
        self.cov_parser = CovParser(config)

    def inc(self, fuzzer_name):
        out_dir = self.config[1]
        queue_dir = os.path.join(out_dir, fuzzer_name, "queue/")

        if os.path.exists(queue_dir):
            prev_size = len(self.dataset)
            testcases = sorted(glob.glob("%s/id:*" % queue_dir))[prev_size:]
            for testcase in testcases:
                data = open(testcase, "rb").read()
                cov = self.cov_parser.parse(data)
                self.dataset.append(data)
                self.max_len = max(len(data), self.max_len)
