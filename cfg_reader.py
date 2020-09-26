#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import angr

proj = angr.Project('./test')
cfg = proj.analyses.CFG(show_progressbar=True)
print(proj.kb.functions)
print(cfg.functions.callgraph)


