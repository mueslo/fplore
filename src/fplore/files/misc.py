# -*- coding: utf-8 -*-
import re

from itertools import zip_longest

from .base import FPLOFile, loads
from .config import FPLOConfig
from ..logging import log


class Error(FPLOFile):
    __fplo_file__ = "+error"
    load_default = True

    @loads('messages')
    def load(self):
        with open(self.filepath, 'r') as f:
            messages = f.read()

        if messages.strip() != "":
            log.warning('+error file not empty:\n{}', messages)

        return messages


class Run(FPLOFile):
    __fplo_file__ = "+run"
    load_default = True

    @loads('data')
    def load(self):
        data = {}
        with open(self.filepath, 'r') as run_file:
            for line in run_file:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()

        return data


class Dens(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.dens"
    load_default = False


class Points(FPLOFile):
    __fplo_file__ = "+points"

    @loads('data')
    def load(self):
        points_file = open(self.filepath, 'r')

        n_points = int(next(points_file).split()[1])
        lines_per_point = 4

        data = []

        for lines in zip_longest(*[points_file] * lines_per_point):
            label_match = re.match("^# ' (.*) '$", lines[0])
            label = label_match.group(1)
            ik = float(lines[1].split()[0])
            data.append((ik, label))

        assert len(data) == n_points
        return data
        # todo: why are there 2 lines, and what's the second number?


class InFile(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.in"

    @loads('_data')
    def load(self):
        data = super(InFile, self).load()['_data']

        self.run.version = (data.header.version.mainversion,
                            data.header.version.subversion)
        log.info("Detected FPLO run with version {}-{}", *self.run.version)
        if float(self.run.version[0]) < 14:
            log.warning("FPLO version <14 is not supported. Things may break.")

        return data


class SymFile(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.sym"
