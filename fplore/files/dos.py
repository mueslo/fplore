# -*- coding: utf-8 -*-

import re
from itertools import zip_longest

import numpy as np

from .base import FPLOFile, loads
from ..logging import log


class DOS(FPLOFile):
    __fplo_file__ = re.compile(r"\+dos\..+")

    @loads('data', 'header')
    def load(self):
        dos_file = open(self.filepath, 'r')

        header = next(dos_file)
        # todo: parse header & filename
        log.debug(header)
        header = {label: int(val) if label != 'nl' else val
                  for label, _, val in
                  zip_longest(*[iter(header.split()[1:])] * 3)}

        data = []
        for line in dos_file:
            ls = line.split()
            if len(ls) == 2:
                data.append(tuple(map(float, line.split())))

        return np.array(data, dtype=[
            ('e', 'f4'),
            ('dos', 'f4'),
        ]), header
