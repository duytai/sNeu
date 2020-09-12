#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import struct

if __name__ == '__main__':
    sancov_files = glob.glob('logs/*.sancov')
    for file in sancov_files:
        print('[+] Read %s' % file)
        with open(file, 'rb') as f:
            data = f.read()
            header = struct.unpack("<Q", data[:8])[0]
            if header != 0xC0BFFFFFFFFFFF64:
                sys.exit('Weird header')
            coverage = dict()
            for i in range(8, len(data), 8):
                offset = data[i : i + 4]
                offset = struct.unpack("<I", offset)[0]
                diff_value = data[i + 4 : i + 8]
                diff_value = struct.unpack("<I", diff_value)[0]
                print("[+] offset: %d" % offset)
                print("[+] diff_value: %d" % diff_value)
                coverage[offset] = diff_value
            for offset, diff_value in coverage.items():
                print("[+] %d\t:\t%d" % (offset, diff_value))
