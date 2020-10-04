import subprocess
import os

class CovParser:
    def __init__(self, config):
        self.config = config

    def parse(self, data):
        bin_dir = self.config[3]
        #  process = subprocess.Popen(
            #  ["%s/target_sneu" % bin_dir],
            #  stdin=subprocess.PIPE,
            #  stdout=subprocess.PIPE,
            #  stderr=subprocess.PIPE,
            #  env=os.environ
        #  )
        #  data = open(testcase, "rb").read()
        #  process.communicate(data)
        #  cov_file = ".logs/%d.cov" % process.pid
        #  cov = open(cov_file, "rb").read()
        #  os.remove(cov_file)
        #  return cov
