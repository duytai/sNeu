import os

class Config:

    def __init__(self, argv):
        for idx in range(0, len(argv), 2):
            opt, val = argv[idx], argv[idx + 1]
            if opt == "-i":
                self.in_dir = val
            if opt == "-o":
                self.out_dir = val
            if opt == "-p":
                self.proj = val
            if opt == "-b":
                self.bin_dir = val
        
        assert self.in_dir, "[x] Require -i"
        assert self.out_dir, "[x] Require -o"
        assert self.proj, "[x] Require -p"
        assert self.bin_dir, "[x] Require -b"

        pwd = os.path.dirname(os.path.realpath(__file__))
        self.pwd = os.path.join(pwd, "../")
        self.target_afl = "target_afl"
        self.target_sneu = "target_sneu"
