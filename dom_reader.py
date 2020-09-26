#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import re

if __name__ == "__main__":
    test_bin = os.path.join(os.getcwd(), "test")

    process = subprocess.Popen(
        ["strings", test_bin],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ
    )
    outs, errs = process.communicate()
    if errs:
        print(errs.decode('utf-8').strip())

    if outs:
        outs = outs.decode('utf-8').strip()
        blocks = re.findall("__sn_[^_]+_\d+_\d+", outs)
        direct_calls = re.findall("__sn_[^_]+_\([^_]+\)_\d+", outs)

        dom_tree = dict()
        func_entry = dict()
        func_call = set()
        vertices = set()

        for block in blocks:
            # parse sn 
            func, from_vertex, to_vertex = block.split("_")[3:]
            func, from_vertex = func[1:-1], int(from_vertex)
            to_vertex = int(to_vertex)
            # init
            if func not in dom_tree:
                dom_tree[func] = set()
                func_entry[func] = from_vertex
            # add vertex
            dom_tree[func].add((from_vertex, to_vertex))
            if func_entry[func] < from_vertex:
                func_entry[func] = from_vertex

        for direct_call in direct_calls:
            caller, callee, from_vertex = direct_call.split("_")[3:]
            caller, callee = caller[1:-1], callee[1:-1]
            from_vertex = int(from_vertex)
            if caller in func_entry:
                if callee in func_entry:
                    func_call.add((from_vertex, func_entry[callee]))

        vertices = set()
        for func in dom_tree:
            tmp_vertices = set()
            for from_vertex, to_vertex in dom_tree[func]:
                tmp_vertices.add(from_vertex)
                tmp_vertices.add(to_vertex)
            print(tmp_vertices)
            tmp_vertices.remove(func_entry[func])
            vertices = set(list(tmp_vertices) + list(vertices))
        #  print(vertices)

        print(func_call)
        print(dom_tree)
        print(func_entry)
