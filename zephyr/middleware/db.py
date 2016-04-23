from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import open
from future import standard_library
standard_library.install_aliases()
from builtins import object

import os
import glob
# import pymongo
# import h5py
import numpy as np
import scipy.io as io
from pygeo.segyread import SEGYFile
import pickle

from .util import compileDict, readini
from .time import BaseTimeSensitive, TimeMachine

ftypeRegex = {
    'vp':       '^%s(?P<iter>[0-9]*)\.vp(?P<freq>[0-9]*\.?[0-9]+)?[^i]*$',
    'qp':       '^%s(?P<iter>[0-9]*)\.qp(?P<freq>[0-9]*\.?[0-9]+)?.*$',
    'vpi':      '^%s(?P<iter>[0-9]*)\.vpi(?P<freq>[0-9]*\.?[0-9]+)?.*$',
    'rho':      '^%s\.rho$',
    'eps2d':    '^%s\.eps2d$',
    'del2d':    '^%s\.del2d$',
    'theta':    '^%s\.theta$',
    'src':      '^%s\.(new)?src(\.avg)?$',
    'grad':     '^%s(?P<iter>[0-9]*)\.gvp[a-z]?(?P<freq>[0-9]*\.?[0-9]+)?.*$',
    'data':     '^%s\.(ut|vz|vx)[ifoOesrcbt]+(?P<freq>[0-9]*\.?[0-9]+).*$',
    'diff':     '^%s\.ud[ifoOesrcbt]+(?P<freq>[0-9]*\.?[0-9]+).*$',
    'wave':     '^%s(?P<iter>[0-9]*)\.(wave|bwave)(?P<freq>[0-9]*\.?[0-9]+).*$',
    'slice':    '^%s\.sl(?P<iter>[0-9]*)',
}

class UtoutWriter(BaseTimeSensitive):
    '''
    AttributeMapper subclass that implements writing frequency-domain
    data to a .utout file.
    '''

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'projnm':       (True,      None,           str),
    }

    def __call__(self, data, fid=slice(None), ftype='utout'):

        ofreqs = self.freqs[fid]
        ofreqs = [(2*np.pi * freq) + self.dampCoeff for freq in ofreqs]
        outfile = '%s.%s'%(self.projnm, ftype)

        nfreq = len(ofreqs)

        if data.ndim != 3:
            raise Exception('Data must be of shape (nrec, nsrc, nfreq)')

        assert data.shape[2] == nfreq
        nrec = data.shape[0]
        nsrc = data.shape[1]

        with io.FortranFile(outfile, 'w') as ff:
            for i, freq in enumerate(ofreqs):
                panel = np.empty((nsrc, nrec+1), dtype=np.complex64)
                panel[:,:1] = freq
                panel[:,1:] = data[:,:,i].T
                ff.write_record(panel.ravel())


class BaseDatastore(object):

    def __init__(self, projnm):

        pass

    @property
    def systemConfig(self):

        raise NotImplementedError


class FullwvDatastore(BaseDatastore):

    def __init__(self, projnm):

        self.projnm = projnm
        inifile = '%s.ini'%projnm

        if not os.path.isfile(inifile):
            raise Exception('Project file %s does not exist'%(inifile,))

        ini = readini(inifile)
        self.ini = ini

        redict = compileDict(projnm, ftypeRegex)

        keepers = {key: {} for key in redict}
        files = glob.glob('*')
        for file in files:
            for key in redict:
                match = redict[key].match(file)
                if match is not None:
                    keepers[key][file] = match.groupdict()
                    break
        self.keepers = keepers

        handled = {}
        for ftype in self.keepers:
            for fn in self.keepers[ftype]:
                handled[fn] = self.handle(ftype, fn)
        self.handled = handled

    @staticmethod
    def sfWrapper(filename):

        sf = SEGYFile(filename)
        return sf

    def handle(self, ftype, filename):

        return self.sfWrapper(filename)

    def __getitem__(self, item):

        if type(item) in {str, unicode}:
            key = item
            sl = slice(None)
        elif type(item) is tuple:
            assert len(item) == 2
            key = item[0]
            sl = item[1]
            assert type(key) is str
            assert (type(sl) is slice) or (type(sl) is int)
        else:
            raise TypeError()

        if key.find(self.projnm) != 0:
            key = self.projnm + key

        if key in self:
            return self.handled[key][sl]
        else:
            raise KeyError(key)

    def __contains__(self, key):

        if key.find(self.projnm) != 0:
            key = self.projnm + key
        return key in self.handled

    def keys(self):
        return list(self.handled.keys())

    def __repr__(self):
        report = {
            'name': self.__class__.__name__,
            'projnm': self.projnm,
            'nfiles': len(self.handled),
        }
        return '<%(name)s(%(projnm)s) comprising %(nfiles)d files>'%report

    @property
    def systemConfig(self):

        transferKeys = {
            'nx':       None,
            'nz':       None,
            'dx':       None,
            'dz':       None,
            'xorig':    None,
            'zorig':    None,
            'freqs':    None,
            'nky':      None,
            'isreg':    'ireg',
            'freqbase': 'freqBase',
        }

        sc = {key if transferKeys[key] is None else transferKeys[key]: self.ini[key] for key in transferKeys}

        sc['tau'] = self.ini['tau'] if abs(np.float(self.ini['tau']) - 999.999) > 1e-2 else np.inf

        sc['freeSurf'] = (
            self.ini['fst'],
            self.ini['fsr'],
            self.ini['fsb'],
            self.ini['fsl'],
        )

        if self.ini['srcs'].shape[1] <=3:
            srcGeom = self.ini['srcs'][:,:2]
            recGeom = self.ini['recs'][:,:2]
        elif self.ini['srcs'].shape[1] == 4:
            srcGeom = self.ini['srcs'][:,::2]
            recGeom = self.ini['recs'][:,::2]
        else:
            raise Exception('Something went wrong!')

        sc['geom'] = {
            'src':      srcGeom,
            'rec':      recGeom,
            'mode':     'fixed',
        }

        fn = '.vp'
        if fn in self:
            sc['c'] = self[fn].T

        fn = '.qp'
        if fn in self:
            sc['Q'] = 1./self[fn].T

        fn = '.rho'
        if fn in self:
            sc['rho'] = self[fn].T

        fn = '.eps2d'
        if fn in self:
            sc['eps'] = self[fn].T

        fn = '.del2d'
        if fn in self:
            sc['delta'] = self[fn].T

        fn = '.theta'
        if fn in self:
            sc['theta'] = self[fn].T

        fn = '.src'
        if fn in self:
            src = self[fn]
            nsrc = srcGeom.shape[0]
            tm = TimeMachine(sc)
            if src.shape[0] != 1 and src.shape[0] != nsrc:
                print('Source nsrc does not match project nsrc; using first term for all sources')
                src = src[:0,:]
            assert src.shape[1] == tm.ns, 'Source ns does not match computed ns'
            sterms = tm.dft(src)
            sc['sterms'] = sterms[:,1:tm.ns//2+1].T

        sc['projnm'] = self.projnm

        return sc

    def dataFiles(self, ftype):

        dKeep = self.keepers['data']
        fns = [fn for fn in dKeep if fn.find(ftype) > -1]
        ffreqs = [float(dKeep[fn]['freq']) for fn in fns]
        order = np.argsort(ffreqs)
        fns = [fns[i] for i in order]
        ffreqs = [ffreqs[i] for i in order]

        return fns, ffreqs

    def spoolData(self, fid=slice(None), ftype='utobs'):

        ifreqs = self.ini['freqs'][fid]
        fns, ffreqs = self.dataFiles(ftype)
        sffreqs = ['%0.3f'%freq for freq in ffreqs]
        try:
            finds = [sffreqs.index('%0.3f'%freq) for freq in ifreqs]
        except ValueError as e:
            raise ValueError('Could not find data from all requested frequencies: %s'%e)

        for fi in finds:
            fdata = self[fns[fi]]
            yield fdata[::2].T + 1j*fdata[1::2].T

    def utoutWrite(self, data, fid=slice(None), ftype='utout'):

        utow = UtoutWriter(self.systemConfig)
        utow(data, fid, ftype)

    # def utoutRead(self, fid=slice(None), ftype='utout')

        # write(50) (omega,(utest(ir,isrc),ir=1,nr),isrc=1,ns)

    # def toHDF5(self, filename):


class FlatDatastore(BaseDatastore):

    def __init__(self, projnm):

        infile = '%s.py'%(projnm,)
        with open(infile, 'r') as fp:
            contents = fp.read()

        #execfile(infile)
        exec(contents, locals())

        self.systemConfig = systemConfig

    @property
    def systemConfig(self):
        return self._systemConfig
    @systemConfig.setter
    def systemConfig(self, value):
        self._systemConfig = value


class PickleDatastore(BaseDatastore):

    def __init__(self, projnm):

        infile = '%s.pickle'%(projnm,)
        with open(infile, 'rb') as fp:
            unp = pickle.Unpickler(fp)
            systemConfig = unp.load()

        self.systemConfig = systemConfig


# class HDF5Datastore(BaseDatastore):

#     def __init__(self, projnm):

#         self.projnm = projnm
#         try:
#             h5file = glob.glob('%s.h*5'%projnm)[0]
#         except IndexError:
#             h5file = '%s.hdf5'%projnm
#             # raise Exception('Project database %(projnm)s.h5 or %(projnm)s.hdf5 does not exist'%{'projnm': projnm})

#         self.db = h5py.File(h5file)


#     pass

# class MongoDBDatastore(BaseDatastore):

#     def __init__(self, mongoURI=None):

#         if mongoURI is None:
#             mongoURI = os.environ.get('MONGO_PORT', 'mongo://localhost:27017').replace('tcp', 'mongodb')

#         self.mc = pymongo.MongoClient(mongoURI)
#         self.db = self.mc.zephyr

#     pass
