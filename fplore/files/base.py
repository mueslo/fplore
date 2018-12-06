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

        loader = getattr(cls, '_load', None)
        if loader and hasattr(loader, '_cache_attrs'):
            setattr(cls, '_load',
                    cache_decorator(loader, cls.__name__, loader._cache_attrs))


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
            self._load()
        except KeyError:
            raise FPLOFileException(
                "FPLO file class {} has no '_load' function".format(
                    self.__name__))

        self.is_loaded = True

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


def cache(*attrs):
    def decorator(f):
        f._cache_attrs = tuple(attr for attr in attrs)
        return f
    return decorator


def cache_decorator(f, classname, attrs):
    if len(attrs) != 1:
        raise NotImplementedError
    attr = attrs[0]

    def _load(self):
        path, filename = os.path.split(self.filepath)
        mtime = str(os.path.getmtime(self.filepath))
        fsize = str(os.path.getsize(self.filepath))

        vals = (mtime, fsize)
        # hashpath = hashlib.sha1(os.path.abspath(self.filepath)).hexdigest()
        hashed = hashlib.sha1(" ".join(vals).encode('utf8')).hexdigest()

        cachedir = "{}/.cache".format(path)
        cachefile = "{}.{}-{}.npy".format(classname, attr, hashed)
        cachepath = os.path.join(cachedir, cachefile)

        # load from cache
        if os.path.isfile(cachepath):
            setattr(self, attr, np.load(cachepath))
            getattr(self, attr).flags.writeable = False
            log.info('Loaded {} from cache ({}).', filename, cachefile)
        else:
            f(self)
            getattr(self, attr).flags.writeable = False
            if not os.path.isdir(cachedir):
                os.mkdir(cachedir)
            np.save(cachepath, getattr(self, attr))
            log.debug('Created cache {}.', cachefile)

    return _load
