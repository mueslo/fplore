# -*- coding: utf-8 -*-

import re
from itertools import zip_longest

import numpy as np

from .base import FPLOFile, loads
from ..logging import log


class DOS(FPLOFile):
    __fplo_file__ = re.compile(r"\+i?l?n?dos\..+")

    @loads('data', 'header')
    def load(self):
        dos_file = open(self.filepath, 'r')

        sections = dict()

        for line in dos_file:
            if line.startswith('#'):
                header = line
                log.debug(header)
                header = {label: int(val) if label != 'nl' else val
                          for label, _, val in
                          zip_longest(*[iter(header.split()[1:])] * 3)}
                log.debug(header)
                header = tuple(header.items())
                log.debug(header)
                sections[header] = []
            else:
                ls = line.split()
                if len(ls) == 2:
                    sections[header].append(tuple(map(float, line.split())))

        headers = list(sections.keys())
        data = [np.array(sections[hdr], dtype=[('e', 'f4'), ('dos', 'f4')]) for hdr in sections.keys()]

        return data, headers
