from __future__ import print_function

import numpy as np
import scipy.sparse as sp
from ..backend import BaseModelDependent
import SimPEG

class HelmFields(SimPEG.Fields.Fields):
    """Fancy Field Storage for frequency domain problems
        u[:,'phi', freqInd] = phi
        print u[src0,'phi']
    """

    knownFields = {'u': 'N'}
    aliasFields = None
    dtype = np.complex128

    def startup(self):
        pass

    def _storageShape(self, loc):
        nP = {'CC': self.mesh.nC,
              'N':  self.mesh.nN,
              'F':  self.mesh.nF,
              'E':  self.mesh.nE}[loc]
        nSrc = self.survey.nSrc
        nFreq = self.survey.nfreq
        return (nP, nSrc, nFreq)

    def _indexAndNameFromKey(self, key, accessType):
        if type(key) is not tuple:
            key = (key,)
        if len(key) == 1:
            key += (None,)
        if len(key) == 2:
            key += (slice(None,None,None),)

        assert len(key) == 3, 'must be [Src, fieldName, freqs]'

        srcTestList, name, freqInd = key

        name = self._nameIndex(name, accessType)
        srcInd = self._srcIndex(srcTestList)

        return (srcInd, freqInd), name

    def _correctShape(self, name, ind, deflate=False):
        srcInd, freqInd = ind
        if name in self.knownFields:
            loc = self.knownFields[name]
        else:
            loc = self.aliasFields[name][1]
        nP, total_nSrc, total_nF = self._storageShape(loc)
        nSrc = np.ones(total_nSrc, dtype=bool)[srcInd].sum()
        nF  = np.ones(total_nF, dtype=bool)[freqInd].sum()
        shape = nP, nSrc, nF
        if deflate:
             shape = tuple([s for s in shape if s > 1])
        if len(shape) == 1:
            shape = shape + (1,)
        return shape
    
    def _setField(self, field, val, name, ind):
        srcInd, freqInd = ind
        shape = self._correctShape(name, ind)
        if SimPEG.Utils.isScalar(val):
            field[:,srcInd,freqInd] = val
            return
        if val.size != np.array(shape).prod():
            print('val.size: %r'%(val.size,))
            print('np.array(shape).prod(): %r'%(np.array(shape).prod(),))
            raise ValueError('Incorrect size for data.')
        correctShape = field[:,srcInd,freqInd].shape
        field[:,srcInd,freqInd] = val.reshape(correctShape, order='F')
    
    def _getField(self, name, ind):
        srcInd, freqInd = ind

        if name in self._fields:
            out = self._fields[name][:,srcInd,freqInd]
        else:
            # Aliased fields
            alias, loc, func = self.aliasFields[name]
            if type(func) is str:
                assert hasattr(self, func), 'The alias field function is a string, but it does not exist in the Fields class.'
                func = getattr(self, func)
            pointerFields = self._fields[alias][:,srcInd,freqInd]
            pointerShape = self._correctShape(alias, ind)
            pointerFields = pointerFields.reshape(pointerShape, order='F')

            freqII = np.arange(self.survey.nfreq)[freqInd]
            srcII  = np.array(self.survey.srcList)[srcInd]
            srcII  = srcII.tolist()

            if freqII.size == 1:
                pointerShapeDeflated = self._correctShape(alias, ind, deflate=True)
                pointerFields = pointerFields.reshape(pointerShapeDeflated, order='F')
                out = func(pointerFields, srcII, freqII)
            else: #loop over the frequencies
                nF = pointerShape[2]
                out = range(nF)
                for i, FIND_i in enumerate(freqII):
                    fieldI = pointerFields[:,:,i]
                    if fieldI.shape[0] == fieldI.size:
                        fieldI = SimPEG.Utils.mkvc(fieldI, 2)
                    out[i] = func(fieldI, srcII, FIND_i)
                    if out[i].ndim == 1:
                        out[i] = out[i][:,np.newaxis,np.newaxis]
                    elif out[i].ndim == 2:
                        out[i] = out[i][:,:,np.newaxis]
                out = np.concatenate(out, axis=2)

        shape = self._correctShape(name, ind, deflate=True)
        return out.reshape(shape, order='F')
    
    def __repr__(self):
        
        shape = self._storageShape('N')        
        attrs = {
            'name':     self.__class__.__name__,
            'id':       id(self),
            'nFields':  len(self.knownFields) + len(self.aliasFields),
            'nN':       shape[0],
            'nSrc':     shape[1],
            'nFreq':    shape[2],
        }
        
        return '<%(name)s container at 0x%(id)x: %(nFields)d fields, with N shape (%(nN)d, %(nSrc)d, %(nFreq)d)>'%attrs
