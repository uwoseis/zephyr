
import os
import glob
# import pymongo
# import h5py
import numpy as np
from pygeo.segyread import SEGYFile
import cPickle

from .util import compileDict, readini

ftypeRegex = {
    'vp':       '^%s(?P<iter>[0-9]*)\.vp(?P<freq>[0-9]*\.?[0-9]+)?[^i]*$',
    'qp':       '^%s(?P<iter>[0-9]*)\.qp(?P<freq>[0-9]*\.?[0-9]+)?.*$',
    'vpi':      '^%s(?P<iter>[0-9]*)\.vpi(?P<freq>[0-9]*\.?[0-9]+)?.*$',
    'src':      '^%s\.(new)?src(\.avg)?$',
    'gvp':      '^%s(?P<iter>[0-9]*)\.gvp[a-z]?(?P<freq>[0-9]*\.?[0-9]+)?.*$',
    'utest':    '^%s\.(ut|vz|vx)[ifoOesrcbt]+(?P<freq>[0-9]*\.?[0-9]+).*$',
    'udiff':    '^%s\.ud[ifoOesrcbt]+(?P<freq>[0-9]*\.?[0-9]+).*$',
    'wave':     '^%s(?P<iter>[0-9]*)\.(wave|bwave)(?P<freq>[0-9]*\.?[0-9]+).*$',
    'slice':    '^%s\.sl(?P<iter>[0-9]*)',
}


class BaseDatastore():
    
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

    def sfWrapper(self, filename):

        sf = SEGYFile(filename)
        return sf

    def handle(self, ftype, filename):

        return self.sfWrapper(filename)

    def __getitem__(self, item):

        if type(item) is str:
            key = item
            sl = slice(None)
        elif type(item) is tuple:
            assert len(item) == 2
            key = item[0]
            sl = item[1]
            assert type(key) is str
            assert (type(sl) is slice) or (type(sl) is int)

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
        return self.handled.keys()

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
            'tau':      None,
            'isreg':    'ireg',
        }
        
        sc = {key if transferKeys[key] is None else transferKeys[key]: self.ini[key] for key in transferKeys}
                
        sc['freeSurf'] = (
            self.ini['fst'],
            self.ini['fsr'],
            self.ini['fsb'],
            self.ini['fsl'],
        )
        
        sc['geom'] = {
            'src':      self.ini['srcs'][:,:2],
            'rec':      self.ini['recs'][:,:2],
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
        
        return sc

    # def toHDF5(self, filename):


class FlatDatastore(BaseDatastore):

    pass

class PickleDatastore(BaseDatastore):
    
    pass
    

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
