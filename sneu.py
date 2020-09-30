#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, signal
import sys
import re
import subprocess
import glob

pwd = os.path.dirname(os.path.realpath(__file__))
rustc = os.path.join(pwd, "rustc.py") 

in_dir = ""
out_dir = ""
bin_dir = ""
proj = ""

target_afl = "target_afl"
target_sneu = "target_sneu"

if __name__ == "__main__":
    argv = sys.argv[1:]

    for idx in range(0, len(argv), 2):
        opt, val = argv[idx], argv[idx + 1]
        if opt == "-i":
            in_dir = val
        if opt == "-o":
            out_dir = val
        if opt == "-p":
            proj = val
        if opt == "-b":
            bin_dir = val

    print("[!] in_dir = %s" % in_dir)
    print("[!] out_dir = %s" % out_dir)
    print("[!] bin_dir = %s" % bin_dir)
    print("[!] proj = %s" % proj)

    # Is proj a rust project
    toml = os.path.join(proj, "Cargo.toml")
    if not os.path.exists(toml):
        print("[x] %s is not found" % toml)
        sys.exit()

    ## Detect package name
    toml_str = open(toml, "r").read()
    matches = re.findall("name\s*=\s*\"([^\"]+)\"", toml_str)
    if not len(matches):
        print("[x] package name is not found")
        sys.exit()

    package = matches[0]
    print("[+] rust package: '%s'" % package)

    if not os.path.exists(os.path.join(bin_dir, target_afl)):
        ## Build fuzz-target for afl 
        print("[+] build fuzz-target for AFL")
        os.environ["RUSTC"] = rustc
        os.system("cd %s && cargo clean && cargo build" % proj)

        ## Copy bin target to bin folder
        print("[+] copy %s to %s" % (target_afl, bin_dir))
        os.system("cd %s && cp target/debug/%s %s/%s" % (proj, package, bin_dir, target_afl))
    else:
        print("[+] found %s" % target_afl)

    if not os.path.exists(os.path.join(bin_dir, target_sneu)):
        ## Build fuzz-target for afl 
        print("[+] build fuzz-target for SNEU")
        os.environ["RUSTC"] = rustc
        os.environ["USE_SNEU"] = "1"
        os.system("cd %s && cargo clean && cargo build" % proj)

        ## Copy bin target to bin folder
        print("[+] copy %s to %s" % (target_sneu, bin_dir))
        os.system("cd %s && cp target/debug/%s %s/%s" % (proj, package, bin_dir, target_sneu))
    else:
        print("[+] found %s" % target_sneu)

    ## Run AFL in subprocess
    pid = os.fork()

    if pid > 0:
        ## Run AFL in parent process
        process = subprocess.Popen(
            ["afl-fuzz", "-i", in_dir, "-o", out_dir, "%s/%s" % (bin_dir, target_afl)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ
        )
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip().decode("utf-8"))
        os.kill(process.pid, signal.SIGINT)
    else:
        ## Mutate in child process 
        os.environ["TARGET_AFL"] = "%s/%s" % (bin_dir, target_afl)
        os.environ["TARGET_SNEU"] = "%s/%s" % (bin_dir, target_sneu)
        os.environ["IN_DIR"] = "%s/queue" % out_dir
        subprocess.call("cd %s && ./mutator.py" % pwd, shell=True)
