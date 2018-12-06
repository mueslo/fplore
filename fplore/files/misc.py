# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import itertools
import re

from .base import FPLOFile
from .config import FPLOConfig
from ..logging import log


class Error(FPLOFile):
    __fplo_file__ = "+error"
    load_default = True

    def _load(self):
        self.messages = open(self.filepath, 'r').read()

        if self.messages.strip() != "":
            log.warning('+error file not empty:\n{}', self.messages)


class Run(FPLOFile):
    __fplo_file__ = "+run"
    load_default = True

    def _load(self):
        self.attrs = {}
        with open(self.filepath, 'r') as run_file:
            for line in run_file:
                key, value = line.split(':', 1)
                self.attrs[key.strip()] = value.strip()


class Dens(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.dens"


class Points(FPLOFile):
    __fplo_file__ = "+points"

    def _load(self):
        points_file = open(self.filepath, 'r')

        self.n_points = int(next(points_file).split()[1])
        lines_per_point = 4

        self.data = []

        for lines in itertools.zip_longest(*[points_file] * lines_per_point):
            label_match = re.match("^# ' (.*) '$", lines[0])
            label = label_match.group(1)
            ik = float(lines[1].split()[0])
            self.data.append((ik, label))

        # todo: why are there 2 lines, and what's the second number?


class InFile(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.in"

    def _load(self):
        super(InFile, self)._load()

        self.run.version = (self.header.version.mainversion,
                            self.header.version.subversion)
        log.info("FPLO run with version {}-{}", *self.run.version)


class SymFile(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.sym"
