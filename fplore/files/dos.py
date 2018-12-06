# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import re

import numpy as np

from .base import FPLOFile
from ..logging import log


class DOS(FPLOFile):
    __fplo_file__ = re.compile(r"\+dos\..+")

    def _load(self):
        dos_file = open(self.filepath, 'r')

        header = next(dos_file)
        # todo: parse header & filename
        log.debug(header)

        data = []
        for line in dos_file:
            ls = line.split()
            if len(ls) == 2:
                data.append(tuple(map(float, line.split())))

        self.data = np.array(data, dtype=[
            ('e', 'f4'),
            ('dos', 'f4'),
        ])
