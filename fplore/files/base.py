# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
from six import with_metaclass
import re
from collections import OrderedDict
from contextlib import contextmanager
import hashlib

import numpy as np
from cached_property import cached_property

from ..logging import log

RegexType = type(re.compile(''))


class FPLOFileException(Exception):
    pass


class FPLOFileType(type):
    def __init__(cls, name, bases, attrs):
        def register_loader(filename):
            cls.registry['loaders'][filename] = cls

        fplo_file = getattr(cls, '__fplo_file__', None)

        if fplo_file:
            if isinstance(fplo_file, str):
                register_loader(fplo_file)

            elif isinstance(fplo_file, RegexType):
                cls.registry['loaders_re'][fplo_file] = cls
            else:
                for f in fplo_file:
                    register_loader(f)

        for name, obj in attrs.items():
            if hasattr(obj, '_loader_property'):
                if hasattr(obj, '_disk_cache_attr'):
                    obj = cache_decorator(cls.__name__, name)(obj)
                obj = cached_property(obj)
                cls._loader_property = name

            setattr(cls, name, obj)


class FPLOFile(with_metaclass(FPLOFileType, object)):
    registry = {'loaders': {}, 'loaders_re': OrderedDict()}
    is_loaded = False
    load_default = False

    @classmethod
    def get_file_class(cls, path):
        fname = os.path.basename(path)
        try:
            return cls.registry['loaders'][fname]
        except KeyError:
            for rgx, loader in cls.registry['loaders_re'].items():
                if rgx.fullmatch(fname):
                    return loader
            raise

    @classmethod
    def open(cls, path, load=False, run=None):
        if os.path.isdir(path):
            raise Exception("Not a file.")

        FileClass = cls.get_file_class(path)
        file_obj = FileClass(path, run=run)
        if load or (load is None and cls.load_default):
            file_obj.load()

        return file_obj

    @classmethod
    def load(cls, path):
        cls.open(path, load=True)

    def __load(self):
        if self.is_loaded:
            log.notice('Reloading {}', self.filepath)

        try:
            getattr(self, self._loader_property)
            self.is_loaded = True
        except AttributeError:
            raise FPLOFileException("FPLO file class <{}> has "
                                    "no designated loading function".format(
                                        type(self).__name__))

    def __init__(self, filepath, run=None):
        self.load = self.__load
        self.filepath = filepath
        self.run = run
        # todo: load run if None


@contextmanager
def writeable(var):
    _writeable = var.flags.writeable
    var.flags.writeable = True
    try:
        yield
    finally:
        var.flags.writeable = _writeable


def loader_property(disk_cache=False):
    def decorator(f):
        f._loader_property = True
        if disk_cache:
            f._disk_cache_attr = True
        return f
    return decorator


def cache_decorator(classname, attrname):
    def decorator(f):
        def _load(self):
            # todo numpy memmap for large datasets
            # todo embed version string in hash
            path, filename = os.path.split(self.filepath)
            mtime = str(os.path.getmtime(self.filepath))
            fsize = str(os.path.getsize(self.filepath))

            vals = (mtime, fsize)
            hashed = hashlib.sha1(" ".join(vals).encode('utf8')).hexdigest()

            cachedir = "{}/.cache".format(path)
            cachefile = "{}.{}-{}.npy".format(classname, attrname, hashed)
            cachepath = os.path.join(cachedir, cachefile)

            # load from cache
            if os.path.isfile(cachepath):
                rv = np.load(cachepath)
                rv.flags.writeable = False
                log.info('Loaded {} from cache ({}).', filename, cachefile)
            else:
                if not os.path.isdir(cachedir):
                    os.mkdir(cachedir)
                rv = f(self)
                rv.flags.writeable = False
                np.save(cachepath, rv)
                log.debug('Created cache {}.', cachefile)

            return rv

        _load.__name__ = f.__name__  # so cached_property can do its thang
        return _load
    return decorator
