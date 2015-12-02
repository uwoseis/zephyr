'''
Low-level programming constructs for Zephyr
'''

import warnings
import numpy as np

class ClassProperty(property):
    'Class decorator to enable property behaviour in classes'

    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()


class AMMetaClass(type):
    '''
    Meta class that enables AttributeMapper functionality, including inheritance
    in the dictionary 'initMap'.
    '''
    
    def __new__(mcs, name, bases, attrs):
        'Build a new subclass of AttributeMapper'
        
        baseMaps = [getattr(base, 'initMap', {}) for base in bases][::-1]
        baseMaps.append(attrs.get('initMap', {}))
        
        initMap = {}
        for baseMap in baseMaps:
            initMap.update(baseMap)
            for key in initMap:
                if initMap[key] is None:
                    del(initMap[key])
        
        attrs['initMap'] = initMap
        
        return type.__new__(mcs, name, bases, attrs)
    
    def __call__(cls, *args, **kwargs):
        'Instantiate a subsclass of AttributeMapper'
        
        if len(args) < 1:
            raise TypeError('__init__() takes at least 2 arguments (1 given)')
        systemConfig = args[0]
        
        obj = cls.__new__(cls)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for key in obj.initMap.keys():
                if (key not in systemConfig) and obj.initMap[key][0]:
                    raise ValueError('Class %s requires parameter \'%s\''%(cls.__name__, key))
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
    the metaclass (i.e., AMMetaClass) to determine how to process the
    value corresponding to the same key in systemConfig.
    
    An exception will be raised if the first element in the tuple
    is set to true, but the corresponding key does not exist in the
    systemConfig parameter.
    
    If the second element in the tuple is set to None, the key will be
    defined in the subclass's attribute dictionary as it stands, whereas
    if the second element is a string then that overrides the key.
    
    If the third element in the tuple is set to None, the input argument
    will be set in the subclass dictionary unmodified; however, if the
    third element is a callable then it will be applied to the element
    (e.g., to allow copying and/or typecasting of inputs).
    
    NB: Complex numpy arguments are handled specially: the real part of
    the value is kept and the imaginary part is discarded when they are
    typecast to a float.
    '''
    
    __metaclass__ = AMMetaClass
    
    def __init__(self, systemConfig):
        '''
        AttributeMapper(systemConfig)
        
        Args:
            systemConfig (dict): A set of setup keys
        '''
        
        pass
    
    @ClassProperty
    @classmethod
    def required(cls):
        'Property to return required fields in initMap'

        return set([key for key in cls.initMap if cls.initMap[key][0]])

    @ClassProperty
    @classmethod
    def optional(cls):
        'Property to return optional fields in initMap'

        return set([key for key in cls.initMap if not cls.initMap[key][0]])


class SCFilter(object):
    '''
    A SCFilter class is initialized with a list of classes as arguments.
    For any of those classes that are AttributeMapper subclasses, SCFilter
    determines the required fields in their initMap trees, and the optional
    fields. When called, the SCFilter discards any key in the passed dictionary
    that does not match one of those fields, and raises an error if any of the
    required fields are not present.
    '''
    
    def __init__(self, clslist):
        '''
        SCFilter(clslist)
        
        Args:
            clslist (list): List of classes from which to build the filter
            
        Returns:
            new SCFilter instance
        '''
        
        if not hasattr(clslist, '__contains__'):
            clslist = [clslist]
        
        self.required = reduce(set.union, (cls.required for cls in clslist if issubclass(cls, AMMetaClass)))
        self.optional = reduce(set.union, (cls.optional for cls in clslist if issubclass(cls, AMMetaClass)))
        self.optional.symmetric_difference_update(self.required)
     
    def __call__(self, systemConfig):
        '''
        Args:
            systemConfig (dict): A systemConfig dictionary to filter
        
        Returns:
            dict: Filtered dictionary
        
        Raises:
            ValueError: If a required key is not in the systemConfig
        '''
        
        for key in self.required:
            if key not in systemConfig:
                raise ValueError('%s requires parameter \'%s\''%(cls.__name__, key))
        
        return {key: systemConfig[key] for key in set.union(self.required, self.optional)}


class BaseSCCache(AttributeMapper):
    '''
    Subclass of AttributeMapper that caches (a filtered version of) the
    systemConfig object used to initialize it.
    '''
    
    maskKeys = []
    cacheItems = []
    
    def __init__(self, systemConfig):
        
        super(BaseSCCache, self).__init__(systemConfig)
        self.systemConfig = {key: systemConfig[key] for key in systemConfig if key not in self.maskKeys}
        
    @property
    def systemConfig(self):
        return self._systemConfig
    @systemConfig.setter
    def systemConfig(self, value):
        self._systemConfig = value
        self.clearCache()
    
    def clearCache(self):
        'Clears cached items (e.g., when model is reset).'
        for attr in self.cacheItems:
            if hasattr(self, attr):
                delattr(self, attr)


class BaseModelDependent(AttributeMapper):
    '''
    AttributeMapper subclass that implements model-dependent properties,
    such as grid coordinates and free-surface conditions.
    '''
    
    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'nx':           (True,      None,           np.int64),
        'ny':           (False,     None,           np.int64),
        'nz':           (True,      None,           np.int64),
        'xorig':        (False,     '_xorig',       np.float64),
        'yorig':        (False,     '_xorig',       np.float64),
        'zorig':        (False,     '_zorig',       np.float64),
        'dx':           (False,     '_dx',          np.float64),
        'dy':           (False,     '_dx',          np.float64),
        'dz':           (False,     '_dz',          np.float64),
        'freeSurf':     (False,     '_freeSurf',    tuple),
    }
    
    @property
    def xorig(self):
        return getattr(self, '_xorig', 0.)
    
    @property
    def yorig(self):
        if hasattr(self, 'ny'):
            return getattr(self, '_yorig', 0.)
        else:
            raise AttributeError('%s object is not 3D'%(self.__class__.__name__,))

    @property
    def zorig(self):
        return getattr(self, '_zorig', 0.)
    
    @property
    def dx(self):
        return getattr(self, '_dx', 1.)
    
    @property
    def dy(self):
        if hasattr(self, 'ny'):
            return getattr(self, '_dy', self.dx)
        else:
            raise AttributeError('%s object is not 3D'%(self.__class__.__name__,))
    
    @property
    def dz(self):
        return getattr(self, '_dz', self.dx)
            
    @property
    def freeSurf(self):
        if getattr(self, '_freeSurf', None) is None:
            self._freeSurf = (False, False, False, False)
        return self._freeSurf
    
    @property
    def modelDims(self):
        if hasattr(self, 'ny'):
            return (self.nz, self.ny, self.nx)
        return (self.nz, self.nx)
    
    @property
    def nrow(self):
        return np.prod(self.modelDims)
    
    def toLinearIndex(self, vec):
        '''
        Gets the linear indices in the raveled model coordinates, given
        a <n by 2> array of n x,z coordinates or a <n by 3> array of
        n x,y,z coordinates.
        
        Args:
            vec (np.ndarray): Space coordinate array
        
        Returns:
            np.ndarray: Grid coordinate array
        '''
        
        if hasattr(self, 'ny'):
            return vec[:,0] * self.nx * self.ny + vec[:,1] * self.nx + vec[:,2]
        else:
            return vec[:,0] * self.nx + vec[:,1]

    def toVecIndex(self, lind):
        '''
        Gets the vectorized index for each linear index.
        
        Args:
            lind (np.ndarray): Grid coordinate array
        
        Returns:
            np.ndarray: nD grid coordinate array
        '''
        
        if hasattr(self, 'ny'):
            return np.array([lind / (self.nx * self.ny), np.mod(lind, self.nx), np.mod(lind, self.ny * self.nx)]).T
        else:
            return np.array([lind / self.nx, np.mod(lind, self.nx)]).T


class BaseAnisotropic(BaseModelDependent):

    initMap = {
    #   Argument        Required    Rename as ...   Store as type
        'theta':        (False,     '_theta',       np.float64),
        'eps':          (False,     '_eps',         np.float64),
        'delta':        (False,     '_delta',       np.float64),
    }

    @property
    def theta(self):
        if getattr(self, '_theta', None) is None:
            self._theta = np.zeros((self.nz, self.nx))
            
        if isinstance(self._theta, np.ndarray):
            return self._theta
        else:
            return self._theta * np.ones((self.nz, self.nx), dtype=np.float64)

    @property
    def eps(self):
        if getattr(self, '_eps', None) is None:
            self._eps = np.zeros((self.nz, self.nx))
            
        if isinstance(self._eps, np.ndarray):
            return self._eps
        else:
            return self._eps * np.ones((self.nz, self.nx), dtype=np.float64)

    @property
    def delta(self):
        if getattr(self, '_delta', None) is None:
            self._delta = np.zeros((self.nz, self.nx))
            
        if isinstance(self._delta, np.ndarray):
            return self._delta
        else:
            return self._delta * np.ones((self.nz, self.nx), dtype=np.float64)
