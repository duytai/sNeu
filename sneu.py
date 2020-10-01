#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, signal
import sys
import re
import subprocess
import glob

pwd = os.path.dirname(os.path.realpath(__file__))
rustc = os.path.join(pwd, "rustc.py") 
fuzzer = os.path.join(pwd, "fuzzer")

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

    assert in_dir, "[x] require -i"
    assert out_dir, "[x] require -o"
    assert proj, "[x] require -p"
    assert bin_dir, "[x] require -b"

    # Is proj a rust project
    toml = os.path.join(proj, "Cargo.toml")
    if not os.path.exists(toml):
        print("[x] %s is not found" % toml)
        sys.exit()

    ## Detect package name
    toml_str = open(toml, "r").read()
    matches = re.findall("name\s*=\s*\"([^\"]+)\"", toml_str)
    if not len(matches):
        print("[x] Package name is not found")
        sys.exit()

    package = matches[0]
    print("[+] Rust package: '%s'" % package)

    if not os.path.exists(os.path.join(bin_dir, target_afl)):
        ## Build fuzz-target for afl 
        print("[+] Build fuzz-target for AFL")
        os.environ["RUSTC"] = rustc
        os.system("cd %s && cargo clean && cargo build" % proj)

        ## Copy bin target to bin folder
        print("[+] Copy %s to %s" % (target_afl, bin_dir))
        os.system("cd %s && cp target/debug/%s %s/%s" % (proj, package, bin_dir, target_afl))
    else:
        print("[+] Found %s" % target_afl)

    if not os.path.exists(os.path.join(bin_dir, target_sneu)):
        ## Build fuzz-target for afl 
        print("[+] Build fuzz-target for SNEU")
        os.environ["RUSTC"] = rustc
        os.environ["USE_SNEU"] = "1"
        os.system("cd %s && cargo clean && cargo build" % proj)

        ## Copy bin target to bin folder
        print("[+] Copy %s to %s" % (target_sneu, bin_dir))
        os.system("cd %s && cp target/debug/%s %s/%s" % (proj, package, bin_dir, target_sneu))
    else:
        print("[+] Found %s" % target_sneu)

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
        pid = os.fork()
        if pid > 0:
            subprocess.call("%s %s/target_afl" % (fuzzer, bin_dir), shell=True)
        else:
            pass
            #  subprocess.call("cd %s && ./mutator.py -b %s -o %s" % (pwd, bin_dir, out_dir), shell=True)
