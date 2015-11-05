
import warnings
import numpy as np

class AMMetaClass(type):
    
    def __new__(mcs, name, bases, attrs):
        
        baseMaps = [getattr(base, 'initMap', {}) for base in bases][::-1]
        baseMaps.append(attrs.get('initMap', {}))
        
        initMap = {}
        for baseMap in baseMaps:
            initMap.update(baseMap)
        
        attrs['initMap'] = initMap
        
        return type.__new__(mcs, name, bases, attrs)
    
    def __call__(cls, *args, **kwargs):
        
        if len(args) < 1:
            raise TypeError('__init__() takes at least 2 arguments (1 given)')
        systemConfig = args[0]
        
        obj = cls.__new__(cls)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for key in obj.initMap.keys():
                if (key not in systemConfig) and obj.initMap[key][0]:
                    raise Exception('Class %s requires parameter \'%s\'!'%(cls.__name__, key))
                if key in systemConfig:
                    if obj.initMap[key][2] is None:
                        typer = lambda x: x
                    else:
                        def typer(x):
                            newtype = obj.initMap[key][2]
                            try:
                                return obj.initMap[key][2](x)
                            except TypeError:
                                if np.iscomplex(x) and issubclass(newtype, np.floating):
                                    return typer(x.real)
                                raise
                                
                    if obj.initMap[key][1] is None:
                        setattr(obj, key, typer(systemConfig[key]))
                    else:
                        setattr(obj, obj.initMap[key][1], typer(systemConfig[key]))
        
        obj.__init__(*args, **kwargs)
        return obj


class AttributeMapper(object):
    '''
    An AttributeMapper subclass defines a dictionary initMap, which
    includes keys for mappable inputs expected from the systemConfig
    parameter. The dictionary takes the form:
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'c':            (True,      '_c',           np.complex128),
        'rho':          (False,     '_rho',         np.float64),
        'freq':         (True,      None,           np.complex128),
        'dx':           (False,     '_dx',          np.float64),
        'dz':           (False,     '_dz',          np.float64),
        'nx':           (True,      None,           np.int64),
        'nz':           (True,      None,           np.int64),
        'freeSurf':     (False,     '_freeSurf',    list),
    }
    
    Each value in the dictionary is a tuple, which is interpreted by
    the baseclass (i.e., AttributeMapper) to determine how to process
    the value corresponding to the same key in systemConfig.
    
    An exception will be raised if the first element in the tuple
    is set to true, but the corresponding key does not exist in the
    systemConfig parameter.
    
    If the second element in the tuple is set to None, the key will be
    assigned to the subclass's attribute dictionary unmodified, whereas
    if the second element is a string then that will be the assigned key.
    
    If the third element in the tuple is set to None, the input argument
    will be set unmodified; however, if the third element is a class
    then it will be applied to the element (e.g., to allow typecasting).
    
    NB: Complex numpy arguments are handled specially, and the real part
    of their value is returned when they are typecast to a float.
    '''
    
    __metaclass__ = AMMetaClass
    
    def __init__(self, systemConfig):
        pass

    
class BaseDiscretization(AttributeMapper):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'c':            (True,      '_c',           np.complex128),
        'rho':          (False,     '_rho',         np.float64),
        'freq':         (True,      None,           np.complex128),
        'dx':           (False,     '_dx',          np.float64),
        'dz':           (False,     '_dz',          np.float64),
        'nx':           (True,      None,           np.int64),
        'nz':           (True,      None,           np.int64),
        'freeSurf':     (False,     '_freeSurf',    list),
    }
    
    @property
    def c(self):
        if isinstance(self._c, np.ndarray):
            return self._c
        else:
            return self._c * np.ones((self.nz, self.nx), dtype=np.complex128)
    
    @property
    def rho(self):
        if getattr(self, '_rho', None) is None:
            self._rho = 310. * self.c**0.25 
            
        if isinstance(self._rho, np.ndarray):
            return self._rho
        else:
            return self._rho * np.ones((self.nz, self.nx), dtype=np.float64)
        
    @property
    def dx(self):
        return getattr(self, '_dx', 1.)
    
    @property
    def dz(self):
        return getattr(self, '_dz', self.dx)
    
    @property
    def freeSurf(self):
        if getattr(self, '_freeSurf', None) is None:
            self._freeSurf = (False, False, False, False)
        return self._freeSurf

class BaseSource(AttributeMapper):
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'xorig':        (False,     '_xorig',       np.float64),
        'zorig':        (False,     '_zorig',       np.float64),
        'dx':           (False,     '_dx',          np.float64),
        'dz':           (False,     '_dz',          np.float64),
        'nx':           (True,      None,           np.int64),
        'nz':           (True,      None,           np.int64),
    }
    
    @property
    def xorig(self):
        return getattr(self, '_xorig', 0.)

    @property
    def zorig(self):
        return getattr(self, '_zorig', 0.)
    
    @property
    def dx(self):
        return getattr(self, '_dx', 1.)
    
    @property
    def dz(self):
        return getattr(self, '_dz', 1.)
