# -*- coding: utf-8 -*-
import sys

import logbook.more
#from logbook.compat import redirect_logging
#redirect_logging()


class StreamHandler(logbook.more.ColorizingStreamHandlerMixin,
                    logbook.StreamHandler):
    pass


sh = StreamHandler(sys.stdout, level='NOTICE', bubble=True)
sh.push_application()

log = logbook.Logger("fplore")