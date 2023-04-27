# -*- coding: utf-8 -*-

import os
from functools import wraps
import re
from collections import OrderedDict
from contextlib import contextmanager
from struct import pack

import numpy as np

from ..logging import log

RegexType = type(re.compile(''))


class FPLOFileException(Exception):
    pass


def loads(*attrs, **kwargs):
    def wrapper(load_orig):
        load_orig._loaded_attrs = attrs
        load_orig._disk_cache = kwargs.get('disk_cache', False)
        load_orig._mem_map = kwargs.get('mem_map', set())
        return load_orig
    return wrapper


def get_cachepath(classname, attrname, filepath):
    path, filename = os.path.split(filepath)
    mtime = os.path.getmtime(filepath)
    fsize = os.path.getsize(filepath)

    try:
        cksum = pack('f', mtime).hex() + pack('I', fsize).hex()
    except AttributeError:  # Py2
        cksum = pack('f', mtime).encode('hex') + pack('I', fsize).encode('hex')

    cachedir = "{}/.cache".format(path)
    cachefile = "{}.{}-{}-{}.npy".format(classname, attrname, filename, cksum)
    return os.path.join(cachedir, cachefile)


def load_wrapper(load_orig):
    """Turns return value into dict with loaded attribute names as keys"""
    loaded_attrs = getattr(load_orig, '_loaded_attrs', None)

    @wraps(load_orig)
    def load(self):
        rv = load_orig(self)
        if loaded_attrs is not None and len(loaded_attrs) > 0:
            if len(loaded_attrs) == 1:
                rv = {loaded_attrs[0]: rv}
            else:
                rv = dict(zip(loaded_attrs, rv))
        return rv

    return load


def load_cache_wrapper(classname, load_orig):
    @wraps(load_orig)
    def load(self):
        try:
            rv = self._load_cache
        except AttributeError:
            loaded_attrs = getattr(load_orig, '_loaded_attrs', None)
            mem_map = getattr(load_orig, '_mem_map', set())
            disk_cache = bool(mem_map) or getattr(load_orig,
                                                  '_disk_cache', False)

            if not disk_cache:
                rv = load_orig(self)
            else:
                rv = {}
                for attrname in loaded_attrs:
                    cachepath = get_cachepath(classname, attrname,
                                              self.filepath)

                    if not os.path.isfile(cachepath):
                        log.debug('Creating cache for {}', classname)
                        if not os.path.isdir(os.path.dirname(cachepath)):
                            os.mkdir(os.path.dirname(cachepath))

                        rv = load_orig(self)

                        for a, v in rv.items():
                            cp = get_cachepath(classname, a, self.filepath)
                            # todo: possibly allow pickle for non-mem-mapped
                            np.save(cp, v, allow_pickle=False)
                            log.debug('Created cache {}.', cp)

                    if attrname in mem_map:
                        attr_rv = np.load(cachepath, mmap_mode='r')
                        log.info('Mem-mapped {} from cache ({}).',
                                 attrname, cachepath)
                    else:
                        attr_rv = np.load(cachepath)
                        attr_rv.flags.writeable = False
                        log.info('Loaded {} from cache ({}).',
                                 attrname, cachepath)

                    rv[attrname] = attr_rv

            self._load_cache = rv
            self.is_loaded = True
        return rv
    return load


def loaded_attr(name):
    def attr(self):
        return self.load()[name]

    attr.__name__ = name
    return attr


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

        load = attrs.get('load', None)
        if load is not None and not isinstance(load, classmethod):
            load = load_wrapper(load)
            setattr(cls, 'load', load_cache_wrapper(name, load))

            for attr in load._loaded_attrs:
                setattr(cls, attr, property(loaded_attr(attr)))

        else:
            log.debug("{} has no explicit loader.", name)


class FPLOFile(object, metaclass=FPLOFileType):
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
                if rgx.match(fname):
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
        return cls.open(path, load=True)

    def __init__(self, filepath, run=None):
        self.filepath = filepath
        self.run = run
        # todo: load run if None

    def __repr__(self):
        if self.run:
            args = "'{}', run={}".format(
                os.path.basename(self.filepath), repr(self.run))
        else:
            args = "'{}'".format(self.filepath)
        return "{}({})".format(type(self).__name__, args)


@contextmanager
def writeable(var):
    _writeable = var.flags.writeable
    var.flags.writeable = True
    try:
        yield
    finally:
        var.flags.writeable = _writeable
