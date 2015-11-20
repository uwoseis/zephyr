import h5py
import numpy as np
import os

class LeafNode(object):
    
    def __init__(self, arr, attrs={}):
        
        if isinstance(arr, h5py.Dataset):
            self._dataset = arr
            self.arr = arr
            self.attrs = arr.attrs
            
        else:
            self.arr = arr
            self.attrs = attrs
        
    def __toh5ds__(self, group, name):

        ds = group.create_dataset(None, data=self.arr)
        ds.attrs['__class__'] = self.__class__.__name__
        
        for key in self.attrs:
            ds.attrs[key] = self.attrs[key]

        return ds
    
    def __getitem__(self, key):
        
        return self.arr[key]
    
    def __setitem__(self, key, value):
        
        self.arr[key] = value

    def inflate(self):
        
        if getattr(self, '_dataset', None) is None:
            raise Exception('Cannot inflate a LeafNode without a backing dataset!')
        
        self.attrs = {key: self._dataset.attrs[key] for key in self._dataset.attrs}
        self.arr = self._dataset[:]
    
    def deflate(self):
        
        if getattr(self, '_dataset', None) is None:
            raise Exception('Cannot deflate a LeafNode without a backing dataset!')
        
        if self.arr is not getattr(self, '_dataset', None):
            self._dataset[:] = self.arr[:]
            
            for key in self.attrs:
                self._dataset.attrs[key] = self.attrs[key]
            
            self.attrs = self._dataset.attrs
            self.arr = self._dataset

class InternalNode(object):
    
    def __init__(self, items, attrs={}):
        
        if isinstance(items, h5py.Group):
            self._group = items
            self.items = items
            self.attrs = items.attrs
            
        else:
            self.items = items
            self.attrs = attrs

    def __toh5ds__(self, group, name):

        g = group.create_group(None)
        g.attrs['__class__'] = self.__class__.__name__
        
        for key in self.attrs:
            g.attrs[key] = self.attrs[key]
        
        for key in self.items:
            g[key] = self.items[key]

        return g
    
    def __getitem__(self, key):
        
        return self.items[key]
    
    def __setitem__(self, key, value):
        
        self.items[key] = value

    def inflate(self):
        
        if getattr(self, '_group', None) is None:
            raise Exception('Cannot inflate an InternalNode without a backing group!')
        
        self.attrs = {key: self._group.attrs[key] for key in self._group.attrs}
        self.items = {key: Node(self._group[key]) for key in self._group}
        for key in self.items:
            self.items[key].inflate()
        
    def deflate(self):
        
        if getattr(self, '_group', None) is None:
            raise Exception('Cannot deflate an InternalNode without a backing group!')
        
        if self.items is not getattr(self, '_group', None):
            
            for key in self.items:
                try:
                    self.items[key].deflate()
                except AttributeError:
                    pass
                
                if not key in self._group:
                    self._group[key] = self.items[key]
            
            for key in self.attrs:
                self._group.attrs[key] = self.attrs[key]
            
            self.attrs = self._group.attrs
            self.items = self._group

def Node(node):
    
    if getattr(node, 'attrs', None) is not None:
        
        glo = globals()
        cla = node.attrs.get('__class__', None)
        if cla in glo:
            print('Looking up %s by name'%cla)
            return glo[cla](node)
        
    if isinstance(node, h5py.Group):
        print('Falling back to InternalNode')
        return InternalNode(node)
    
    if isinstance(node, h5py.Dataset):
        print('Falling back to LeafNode')
        return LeafNode(node)
    
    else:
        raise Exception('Entry %r is neither a Group nor a Dataset, nor is it handled by an existing class.'%(node,))

class InversionContainer(object):
    
    cfile = None
    
    def __init__(self, filename):
        
        mainGroups = {
            'fields':   'Field data storage',
            'models':   'Model storage',
            'data':     'Data storage',
            'code':     'Code storage',
            'config':   'Configuration',
        }
        
        if os.path.isfile(filename):
            print('Reading %s'%(filename,))
            self.cfile = h5py.File(filename, 'r+')
            
            for path in self.cfile:
                name = os.path.split(path)[-1]
                node = self.cfile[path]
                try:
                    setattr(self, name, Node(node))
                except:
                    pass
        
        else:
            print('Creating %s'%(filename,))
            self.cfile = h5py.File(filename, 'w')
            
            for key in mainGroups:
                group = self.cfile.create_group(key)
                group.attrs['desc'] = mainGroups[key]
                setattr(self, key, InternalNode(group))

class PhysicalPropertyNode(LeafNode):
    
    @property
    def unit(self):
        return self.attrs.get('unit', None)
    @unit.setter
    def unit(self, value):
        self.attrs['unit'] = value
        
class ModelNode(PhysicalPropertyNode):
    pass

class FieldNode(PhysicalPropertyNode):
    pass

class DataNode(LeafNode):
    pass
    