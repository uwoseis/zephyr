from IPython.parallel import Client, parallel, Reference, require, depend, interactive
import numpy as np
import networkx

DEFAULT_MPI = True
MPI_BELLWETHERS = ['PMI_SIZE', 'OMPI_UNIVERSE_SIZE']

def getChunks(problems, chunks=1):
    nproblems = len(problems)
    return (problems[i*nproblems // chunks: (i+1)*nproblems // chunks] for i in range(chunks))

@interactive
def hasSystem(tag):
    global localSystem
    return tag in localSystem

@interactive
def hasSystemRank(tag, wid):
    global localSystem
    global rank
    return (tag in localSystem) and (rank == wid)

def cdSame(rc):
    import os

    dview = rc[:]

    home = os.getenv('HOME')
    cwd = os.getcwd()

    @interactive
    def cdrel(relpath):
        import os
        home = os.getenv('HOME')
        fullpath = os.path.join(home, relpath)
        try:
            os.chdir(fullpath)
        except OSError:
            return False
        else:
            return True

    if cwd.find(home) == 0:
        relpath = cwd[len(home)+1:]
        return all(rc[:].apply_sync(cdrel, relpath))

def adjustMKLVectorization(nt=1):
    try:
        import mkl
    except ImportError:
        pass
    finally:
        mkl.set_num_threads(nt)

class CommonReducer(dict):

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.addcounter = 0
        self.iaddcounter = 0
        self.interactcounter = 0
        self.callcounter = 0

    def __add__(self, other):
        result = CommonReducer(self)
        for key in other.keys():
            if key in result:
                result[key] = self[key] + other[key]
            else:
                result[key] = other[key]

        self.addcounter += 1
        self.interactcounter += 1

        return result

    def __iadd__(self, other):
        for key in other.keys():
            if key in self:
                self[key] += other[key]
            else:
                self[key] = other[key]

        self.iaddcounter += 1
        self.interactcounter += 1

        return self

    def __mul__(self, other):
        result = CommonReducer()
        for key in other.keys():
            if key in self:
                result[key] = self[key] * other[key]

        return result

    def __sub__(self, other):
        result = CommonReducer()
        for key in other.keys():
            if key in self:
                result[key] = self[key] - other[key]

        return result

    def __div__(self, other):
        result = CommonReducer()
        for key in other.keys():
            if key in self:
                result[key] = self[key] / other[key]

        return result

    def sum(self, *args, **kwargs):
        result = CommonReducer()
        for key in self.keys():
            result[key] = self[key].sum(*args, **kwargs)

        return result

    def log(self):
        result = CommonReducer()
        for key in self.keys():
            result[key] = np.log(self[key])

        return result

    def conj(self):
        result = CommonReducer()
        for key in self.keys():
            result[key] = self[key].conj()

        return result

    def real(self):
        result = CommonReducer()
        for key in self.keys():
            result[key] = self[key].real()

        return result

    def imag(self):
        result = CommonReducer()
        for key in self.keys():
            result[key] = self[key].imag()

        return result

    def ravel(self):
        result = CommonReducer()
        for key in self.keys():
            result[key] = self[key].ravel()

        return result

    def reshape(self, *args, **kwargs):
        result = CommonReducer()
        for key in self.keys():
            result[key] = self[key].reshape(*args, **kwargs)

        return result

    def copy(self):

        return CommonReducer(self)

    def __call__(self, key, result):
        if key in self:
            self[key] += result
        else:
            self[key] = result

        self.callcounter += 1
        self.interactcounter += 1


class SystemSolver(object):

    def __init__(self, dispatcher, schedule):

        self.dispatcher = dispatcher
        self.schedule = schedule

        # for key in schedule:
        #     refs = schedule[key]
        #     handler = lambda target, isrcs: self._systemSolve(isrcs, refs['solve'], refs['clear'])
        #     handler.__name__ = key
        #     setattr(self, key, handler.__get__(self, self.__class__))

    # def _systemSolve(self, isrcs, fnRef, clearRef):

    def __call__(self, entry, isrcs):
        
        fnRef = self.schedule[entry]['solve']
        clearRef = self.schedule[entry]['clear']

        dview = self.dispatcher.remote.dview
        lview = self.dispatcher.remote.lview

        chunksPerWorker = getattr(self.dispatcher, 'chunksPerWorker', 1)

        G = networkx.DiGraph()

        mainNode = 'Beginning'
        G.add_node(mainNode)

        # Parse sources
        nsrc = self.dispatcher.nsrc
        if isrcs is None:
            isrcslist = range(nsrc)

        elif isinstance(isrcs, slice):
            isrcslist = range(isrcs.start or 0, isrcs.stop or nsrc, isrcs.step or 1)

        else:
            try:
                _ = isrcs[0]
                isrcslist = isrcs
            except TypeError:
                isrcslist = [isrcs]

        systemsOnWorkers = dview['localSystem.keys()']
        ids = dview['rank']
        tags = set()
        for ltags in systemsOnWorkers:
            tags = tags.union(set(ltags))

        endNodes = {}
        tailNodes = []

        for tag in tags:

            tagNode = 'Head: %d, %d'%tag
            G.add_edge(mainNode, tagNode)

            relIDs = []
            for i in xrange(len(ids)):

                systems = systemsOnWorkers[i]
                rank = ids[i]

                if tag in systems:
                    relIDs.append(i)

            systemJobs = []
            endNodes[tag] = []
            systemNodes = []

            with lview.temp_flags(block=False):
                iworks = 0
                for work in getChunks(isrcslist, int(round(chunksPerWorker*len(relIDs)))):
                    if work:
                        job = lview.apply(fnRef, tag, work)
                        systemJobs.append(job)
                        label = 'Compute: %d, %d, %d'%(tag[0], tag[1], iworks)
                        systemNodes.append(label)
                        G.add_node(label, job=job)
                        G.add_edge(tagNode, label)
                        iworks += 1

            if getattr(self.dispatcher, 'ensembleClear', False): # True for ensemble ending, False for individual ending
                tagNode = 'Wrap: %d, %d'%tag
                for label in systemNodes:
                    G.add_edge(label, tagNode)

                for i in relIDs:

                    rank = ids[i]

                    with lview.temp_flags(block=False, after=systemJobs):
                        job = lview.apply(depend(hasSystemRank, tag, rank)(clearRef), tag)
                        label = 'Wrap: %d, %d, %d'%(tag[0],tag[1], i)
                        G.add_node(label, job=job)
                        endNodes[tag].append(label)
                        G.add_edge(tagNode, label)
            else:

                for i, sjob in enumerate(systemJobs):
                    with lview.temp_flags(block=False, follow=sjob):
                        job = lview.apply(clearRef, tag)
                        label = 'Wrap: %d, %d, %d'%(tag[0],tag[1],i)
                        G.add_node(label, job=job)
                        endNodes[tag].append(label)
                        G.add_edge(systemNodes[i], label)

            tagNode = 'Tail: %d, %d'%tag
            for label in endNodes[tag]:
                G.add_edge(label, tagNode)
            tailNodes.append(tagNode)

        endNode = 'End'
        for node in tailNodes:
            G.add_edge(node, endNode)

        return G

    def wait(self, G):
        self.dispatcher.remote.lview.wait((G.node[wn]['job'] for wn in (G.predecessors(tn)[0] for tn in G.predecessors('End'))))

class RemoteInterface(object):

    def __init__(self, profile=None, MPI=None, nThreads=1):

        if profile is not None:
            pupdate = {'profile': profile}
        else:
            pupdate = {}

        pclient = Client(**pupdate)

        if not cdSame(pclient):
            print('Could not change all workers to the same directory as the client!')

        dview = pclient[:]
        dview.block = True
        dview.clear()

        remoteSetup = '''
        import os'''

        parMPISetup = ''' 
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()''' 

        for command in remoteSetup.strip().split('\n'):
            dview.execute(command.strip())

        dview.scatter('rank', pclient.ids, flatten=True)

        self.e0 = pclient[0]
        self.e0.block = True

        self.useMPI = False
        MPI = DEFAULT_MPI if MPI is None else MPI
        if MPI:
            MPISafe = False

            for var in MPI_BELLWETHERS:
                MPISafe = MPISafe or all(dview["os.getenv('%s')"%(var,)])

            if MPISafe:
                for command in parMPISetup.strip().split('\n'):
                    dview.execute(command.strip())
                ranks = dview['rank']
                reorder = [ranks.index(i) for i in xrange(len(ranks))]
                dview = pclient[reorder]
                dview.block = True
                dview.activate()

                # Set up necessary parts for broadcast-based communication
                self.e0 = pclient[reorder[0]]
                self.e0.block = True
                self.comm = Reference('comm')

            self.useMPI = MPISafe

        self.pclient = pclient
        self.dview = dview
        self.lview = pclient.load_balanced_view()

        self.nThreads = nThreads

        # Generate 'par' object for Problem to grab
        self.par = {
            'pclient':      self.pclient,
            'dview':        self.dview,
            'lview':        self.pclient.load_balanced_view(),
        }

    @property
    def nThreads(self):
        return self._nThreads
    @nThreads.setter
    def nThreads(self, value):
        self._nThreads = value
        self.dview.apply(adjustMKLVectorization, self._nThreads)
    

    def __setitem__(self, key, item):

        if self.useMPI:
            self.e0[key] = item
            code = 'if rank != 0: %(key)s = None\n%(key)s = comm.bcast(%(key)s, root=0)'
            self.dview.execute(code%{'key': key})

        else:
            self.dview[key] = item

    def __getitem__(self, key):

        if self.useMPI:
            code = 'temp_%(key)s = None\ntemp_%(key)s = comm.gather(%(key)s, root=%(root)d)'
            self.dview.execute(code%{'key': key, 'root': 0})
            item = self.e0['temp_%s'%(key,)]
            self.e0.execute('del temp_%s'%(key,))

        else:
            item = self.dview[key]

        return item

    def reduce(self, key, axis=None):

        if self.useMPI:
            code = 'temp_%(key)s = comm.reduce(%(key)s, root=%(root)d)'
            self.dview.execute(code%{'key': key, 'root': 0})

            # if axis is not None:
            #     code = 'temp_%(key)s = temp_%(key)s.sum(axis=%(axis)d)'
            #     self.e0.execute(code%{'key': key, 'axis': axis})

            item = self.e0['temp_%s'%(key,)]
            self.dview.execute('del temp_%s'%(key,))

        else:
            item = reduce(np.add, self.dview[key])

        return item

    def reduceMul(self, key1, key2, axis=None):

        if self.useMPI:
            # Gather
            code_reduce = 'temp_%(key)s = comm.reduce(%(key)s, root=%(root)d)'
            self.dview.execute(code_reduce%{'key': key1, 'root': 0})
            self.dview.execute(code_reduce%{'key': key2, 'root': 0})

            # Multiply
            code_mul = 'temp_%(key1)s%(key2)s = temp_%(key1)s * temp_%(key2)s'
            self.e0.execute(code_mul%{'key1': key1, 'key2': key2})

            # Potentially sum
            if axis is not None:
                code = 'temp_%(key1)s%(key2)s = temp_%(key1)s%(key2)s.sum(axis=%(axis)d)'
                self.e0.execute(code%{'key1': key1, 'key2': key2, 'axis': axis})

            # Pull
            item = self.e0['temp_%(key1)s%(key2)s'%{'key1': key1, 'key2': key2}]

            # Clear
            self.dview.execute('del temp_%s'%(key1,))
            self.dview.execute('del temp_%s'%(key2,))
            self.e0.execute('del temp_%(key1)s%(key2)s'%{'key1': key1, 'key2': key2})

        else:
            item1 = reduce(np.add, self.dview[key1])
            item2 = reduce(np.add, self.dview[key2])
            item = item1 * item2

        return item

    def remoteDifference(self, key1, key2, keyresult):

        if self.useMPI:

            root = 0

            # Gather
            code_reduce = 'temp_%(key)s = comm.reduce(%(key)s, root=%(root)d)'
            self.dview.execute(code_reduce%{'key': key1, 'root': root})
            self.dview.execute(code_reduce%{'key': key2, 'root': root})

            # Difference
            code_difference = '%(keyresult)s = temp_%(key1)s - temp_%(key2)s'
            self.e0.execute(code_difference%{'key1': key1, 'key2': key2, 'keyresult': keyresult})

            # Broadcast
            code = 'if rank != 0: %(key)s = None\n%(key)s = comm.bcast(%(key)s, root=%(root)d)'
            self.dview.execute(code%{'key': keyresult, 'root': root})

            # Clear
            self.e0.execute('del temp_%s'%(key1,))
            self.e0.execute('del temp_%s'%(key2,))

        else:
            item1 = reduce(np.add, self.dview[key1])
            item2 = reduce(np.add, self.dview[key2])

            item = item1 - item2
            self.dview[keyresult] = item

    def remoteOpGatherFirst(self, op, key1, key2, keyresult):

        if self.useMPI:

            root = 0

            # Gather
            code_reduce = 'temp_%(key)s = comm.reduce(%(key)s, root=%(root)d)'
            self.dview.execute(code_reduce%{'key': key1, 'root': root})

            # Difference
            code_difference = '%(keyresult)s = temp_%(key1)s %(op)s %(key2)s'
            self.e0.execute(code_difference%{'op': op, 'key1': key1, 'key2': key2, 'keyresult': keyresult})

            # Broadcast
            code = 'if rank != 0: %(key)s = None\n%(key)s = comm.bcast(%(key)s, root=%(root)d)'
            self.dview.execute(code%{'key': keyresult, 'root': root})

            # Clear
            self.e0.execute('del temp_%s'%(key1,))

        else:
            item1 = reduce(np.add, self.dview[key1])
            item2 = self.e0[key2] # Assumes that any arbitrary worker has this information

            item = eval('item1 %s item2'%(op,))
            self.dview[keyresult] = item

    def remoteDifferenceGatherFirst(self, *args):
        self.remoteOpGatherFirst('-', *args)

    def normFromDifference(self, key):

        code = 'temp_norm%(key)s = (%(key)s * %(key)s.conj()).sum(0).sum(0)'
        self.e0.execute(code%{'key': key})
        code = 'temp_norm%(key)s = {key: np.sqrt(temp_norm%(key)s[key]).real for key in temp_norm%(key)s.keys()}'
        self.e0.execute(code%{'key': key})
        result = CommonReducer(self.e0['temp_norm%s'%(key,)])
        self.e0.execute('del temp_norm%s'%(key,))

        return result
