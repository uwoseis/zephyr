
import warnings
import numpy as np

class AttributeMapper(object):
    
    initMap = {}
    
    def __init__(self, systemConfig):
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for key in self.initMap.keys():
                if (key not in systemConfig) and self.initMap[key][0]:
                    raise Exception('AttributeMapper subclass %s requires parameter \'%s\'!'%(self.__class__.__name__, key))
                if key in systemConfig:
                    if self.initMap[key][2] is None:
                        typer = lambda x: x
                    else:
                        def typer(x):
                            newtype = self.initMap[key][2]
                            try:
                                return self.initMap[key][2](x)
                            except TypeError:
                                if np.iscomplex(x) and issubclass(newtype, np.floating):
                                    return typer(x.real)
                                raise
                                
                    if self.initMap[key][1] is None:
                        setattr(self, key, typer(systemConfig[key]))
                    else:
                        setattr(self, self.initMap[key][1], typer(systemConfig[key]))

class BaseDiscretization(AttributeMapper):
    
    pass

class BaseSource(AttributeMapper):
    
    pass
