import os
import glob

from mod.covparser import CovParser

class DataLoader:

    def __init__(self, config):
        self.config = config
        self.dataset = []
        self.max_len = 0
        self.cov_parser = CovParser(config)
        self.positive = set()
        self.negative = set()
        self.branch_ids = set()
        self.zeros = set()

    def update(self, branch_id, signed_distance):
        if not signed_distance:
            self.zeros.add(branch_id)
        elif signed_distance > 0:
            self.positive.add(branch_id)
        else:
            self.negative.add(branch_id)

    def uncovered_branches(self):
        positive = self.positive - self.negative - self.zeros
        negative = self.negative - self.positive - self.zeros
        zeros = self.zeros - self.positive - self.negative
        uncovered = list(positive) + list(negative) + list(zeros)
        print(uncovered)
        return uncovered

    def inc(self, fuzzer_name):
        queue_dir = os.path.join(self.config.out_dir, fuzzer_name, "queue/")

        branch = {}
        if os.path.exists(queue_dir):
            prev_size = len(self.dataset)
            testcases = sorted(glob.glob("%s/id:*" % queue_dir))[prev_size:]
            for testcase in testcases:
                data = open(testcase, "rb").read()
                cov = self.cov_parser.parse(data)
                for type_size, branch_id, left_value, right_value in cov:
                    self.update(branch_id, left_value - right_value)
                    branch[branch_id] = (type_size, left_value, right_value)
                    self.branch_ids.add(branch_id)
                self.dataset.append((data, branch))
                self.max_len = max(len(data), self.max_len)
