
from .meta import AttributeMapper, BaseModelDependent
import numpy as np

class RichContainer(AttributeMapper):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'fieldTerms':   (False,     '_fieldTerms',  dict),
        'freqs':        (False,     '_freqs',       np.float64),
        'times':        (False,     '_times',       np.float64),
        'torf':         (False,     None,           str),
    }
    
    def __init__(self, systemConfig):
        
        if self.torf == 'time' and not hasattr(self, '_times'):
            raise ValueError('%s objects of the \'time\' domain require a \'times\' parameter'%(self.__class__.__name__,))
        
        if self.torf == 'freq' and not hasattr(self, '_freqs'):
            raise ValueError('%s objects of the \'freq\' domain require a \'freqs\' parameter'%(self.__class__.__name__,))
        
        nterms = len(self.terms)
        dims = (nterms,) + self.supplementalDims
        for term in self.fieldTerms:
            setattr(self, term, np.zeros(dims, dtype=self.fieldTerms[term]))
    
    @property
    def fieldTerms(self):
        return getattr(self, '_fieldTerms', {'pressure': np.complex128})
     
    @property 
    def torf(self):
        return getattr(self, '_torf', 'freq')
    @torf.setter
    def torf(self, value):
        value = value.lower()
        if value in ('time', 'freq'):
            self._torf = value
        else:
            raise NotImplementedError('Parameter \'torf\' cannot store value %s'%(value,))
    
    @property
    def freqs(self):
        if hasattr(self, '_freqs'):
            return self._freqs
        else:
            raise AttributeError
        
    @property
    def times(self):
        if hasattr(self, '_times'):
            return self._times
        else:
            raise AttributeError
    
    @property
    def terms(self):
        if self.torf == 'freq':
            return self.freqs
        elif self.torf == 'time':
            return self.times
        else:
            raise AttributeError
    
    @property
    def supplementalDims(self):
        raise NotImplementedError
    
    @property
    def indices(self):
        raise NotImplementedError

class Field(RichContainer,BaseModelDependent):
    
#    initMap = {
#    #   Argument        Required    Rename as ...   Store as type
#        'nx':           (True,      None,           np.int64),
#        'ny':           (False,     None,           np.int64),
#        'nz':           (True,      None,           np.int64),
#        'xorig':        (False,     '_xorig',       np.float64),
#        'yorig':        (False,     '_xorig',       np.float64),
#        'zorig':        (False,     '_zorig',       np.float64),
#        'dx':           (False,     '_dx',          np.float64),
#        'dy':           (False,     '_dx',          np.float64),
#        'dz':           (False,     '_dz',          np.float64),
#    }
    
    @property
    def supplementalDims(self):
        if hasattr(self, 'ny'):
            return (self.nz, self.ny, self.nx)
        return (self.nz, self.nx)
    
    @property
    def indices(self):
        if hasattr(self, 'ny'):
            return (self.terms,
                    np.arange(self.zorig, self.zorig + self.dz * (self.nz - 1), self.dz),
                    np.arange(self.yorig, self.yorig + self.dy * (self.ny - 1), self.dy),
                    np.arange(self.xorig, self.xorig + self.dx * (self.nx - 1), self.dx),
                    )
        else:
            return (self.terms,
                    np.arange(self.zorig, self.zorig + self.dz * (self.nz - 1), self.dz),
                    np.arange(self.xorig, self.xorig + self.dx * (self.nx - 1), self.dx),
                    )
        
#    @property
#    def xorig(self):
#        return getattr(self, '_xorig', 0.)
#    
#    @property
#    def yorig(self):
#        if hasattr(self, 'ny'):
#            return getattr(self, '_yorig', 0.)
#        else:
#            raise AttributeError('%s object is not 3D'%(self.__class__.__name__,))
#
#    @property
#    def zorig(self):
#        return getattr(self, '_zorig', 0.)
#    
#    @property
#    def dx(self):
#        return getattr(self, '_dx', 1.)
#    
#    @property
#    def dy(self):
#        if hasattr(self, 'ny'):
#            return getattr(self, '_dy', 1.)
#        else:
#            raise AttributeError('%s object is not 3D'%(self.__class__.__name__,))
#    
#    @property
#    def dz(self):
#        return getattr(self, '_dz', 1.)

class Data(RichContainer):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'nsrc':         (True,      None,           np.int64),
        'nrec':         (True,      None,           np.int64),
    }
    
    @property
    def supplementalDims(self):
        return (self.nsrc, self.nrec)
    
    @property
    def indices(self):
        return (self.terms,
                np.range(self.nsrc),
                np.range(self.nrec),
                )
