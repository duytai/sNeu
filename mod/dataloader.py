import os
import glob
import random
import numpy as np
import torch

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

    def create_dataset(self):
        ret = []
        branch_ids = self.uncovered_branch_ids()
        for data, branch in self.dataset:
            label = np.zeros(len(branch_ids))
            input = np.zeros(self.max_len)
            for idx, branch_id in enumerate(branch_ids):
                if branch_id in branch:
                    left_value, right_value = branch[branch_id][1:]
                    label[idx] = abs(left_value - right_value)
            for idx, val in enumerate(bytearray(data)):
                input[idx] = val
            input = (input - 128.0) / 128.0 # [-1, 1]
            input = torch.tensor(input).float()
            label = np.clip(label / 255.0, 0, 1.0) # [0, 1]
            label = torch.tensor(label).float()
            ret.append((input, label))
        return ret

    def uncovered_branch_ids(self):
        positive = self.positive - self.negative - self.zeros
        negative = self.negative - self.positive - self.zeros
        zeros = self.zeros - self.positive - self.negative
        uncovered = list(positive) + list(negative) + list(zeros)
        return uncovered

    def inc(self, fuzzer_name):
        queue_dir = os.path.join(self.config.out_dir, fuzzer_name, "queue/")
        if os.path.exists(queue_dir):
            prev_size = len(self.dataset)
            testcases = sorted(glob.glob("%s/id:*" % queue_dir))[prev_size:]
            for testcase in testcases:
                data = open(testcase, "rb").read()
                cov = self.cov_parser.parse(data)
                branch = {}
                for type_size, branch_id, left_value, right_value in cov:
                    self.update(branch_id, left_value - right_value)
                    branch[branch_id] = (type_size, left_value, right_value)
                    self.branch_ids.add(branch_id)
                self.dataset.append((data, branch))
                self.max_len = max(len(data), self.max_len)
