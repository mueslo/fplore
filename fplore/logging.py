# -*- coding: utf-8 -*-
import sys

import logbook.more


class StreamHandler(logbook.more.ColorizingStreamHandlerMixin,
                    logbook.StreamHandler):
    pass


sh = StreamHandler(sys.stdout, level='NOTICE', bubble=True)
sh.push_application()

log = logbook.Logger("fplore")
