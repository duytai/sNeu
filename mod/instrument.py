import os
import re

class Instrument:

    def __init__(self, config):
        self.config = config
        self.build_targets()

    def build_targets(self):
        ## Is proj a rust project
        toml = os.path.join(self.config.proj, "Cargo.toml")
        if not os.path.exists(toml):
            print("[x] %s is not found" % toml)
            sys.exit(0)

        ## Detect package name
        toml_str = open(toml, "r").read()
        matches = re.findall("name\s*=\s*\"([^\"]+)\"", toml_str)
        if not len(matches):
            print("[x] Package name is not found")
            sys.exit(0)

        package = matches[0]
        print("[+] Rust package: '%s'" % package)

        if not os.path.exists(os.path.join(self.config.bin_dir, self.config.target_afl)):
            ## Build fuzz-target for afl 
            print("[+] Build fuzz-target for AFL")
            os.environ["RUSTC"] = os.path.join(self.config.pwd, "rustc.py")  
            os.system("cd %s && cargo clean && cargo build" % self.config.proj)

            ## Copy bin target to bin folder
            print("[+] Copy %s to %s" % (self.config.target_afl, self.config.bin_dir))
            os.system("cp %s/target/debug/%s %s/%s" % (self.config.proj, package, self.config.bin_dir, self.config.target_afl))
        else:
            print("[+] Found %s" % self.config.target_afl)

        if not os.path.exists(os.path.join(self.config.bin_dir, self.config.target_sneu)):
            ## Build fuzz-target for afl 
            print("[+] Build fuzz-target for SNEU")
            os.environ["RUSTC"] = os.path.join(self.config.pwd, "rustc.py") 
            os.environ["USE_SNEU"] = "1"
            os.system("cd %s && cargo clean && cargo build" % self.config.proj)

            ## Copy bin target to bin folder
            print("[+] Copy %s to %s" % (self.config.target_sneu, self.config.bin_dir))
            os.system("cp %s/target/debug/%s %s/%s" % (self.config.proj, package, self.config.bin_dir, self.config.target_sneu))
        else:
            print("[+] Found %s" % self.config.target_sneu)
