from __future__ import absolute_import

__version__ = "0.1a"
__author__ = "Johannes Falke <johannesfalke@gmail.com>"
__copyright__ = "Copyright (C) 2018 Johannes Falke"
__license__ = "GNU General Public License v3"

import sys

import logbook.more
#from logbook.compat import redirect_logging
#redirect_logging()


class StreamHandler(logbook.more.ColorizingStreamHandlerMixin,
                    logbook.StreamHandler):
    pass


sh = StreamHandler(sys.stdout, level='DEBUG', bubble=True)
sh.push_application()

log = logbook.Logger("fplore")
