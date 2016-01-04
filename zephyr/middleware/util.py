
import numpy as np

# Code in str2bool and readini pulled from github.com/bsmithyman/pygeo/pygeo/fullpy.py
# Original code is licensed LGPL; I am explicitly re-licensing this under the umbrella
# of the windtunnel MIT license. -- Brendan Smithyman

def str2bool(v):
    '''
    Converts certain string values to a boolean.
    '''

    return v.lower() in ("yes", "true", "t", "1")

def readini (infile):
    '''
    Reads (2.5-D) omega ini file of a given filename.
    '''

    f = open(infile, 'r')
    lines = f.readlines()
    f.close()

    settingsdict = {}

    lsplit = lines[1].strip().split()
    settingsdict['comment'] = int(lsplit[0])
    settingsdict['lessfiles'] = str2bool(lsplit[1])

    lsplit = lines[3].strip().split()
    settingsdict['nx'] = int(lsplit[0])
    settingsdict['nz'] = int(lsplit[1])
    settingsdict['dx'] = float(lsplit[2])
    settingsdict['dz'] = float(lsplit[3])
    settingsdict['xorig'] = float(lsplit[4])
    settingsdict['zorig'] = float(lsplit[5])

    lsplit = lines[5].replace('\'','').strip().split()
    settingsdict['inv'] = str2bool(lsplit[0])
    settingsdict['datain'] = lsplit[1]
    settingsdict['dataout'] = lsplit[2]
    settingsdict['waveout'] = int(lsplit[3])
    settingsdict['usescratch'] = str2bool(lsplit[4])
    settingsdict['nom'] = int(lsplit[5])
    settingsdict['nsam'] = int(lsplit[6])
    settingsdict['tau'] = float(lsplit[7])
    settingsdict['nftout'] = int(lsplit[8])
    settingsdict['tau'] = float(lsplit[7])

    lsplit = lines[7].replace('\'','').strip().split()
    settingsdict['we'] = lsplit[0]
    settingsdict['param'] = int(lsplit[1])
    settingsdict['nky'] = int(lsplit[2])
    settingsdict['method'] = int(lsplit[3])
    settingsdict['vmin'] = float(lsplit[4])
    settingsdict['deltatt'] = float(lsplit[5])
    settingsdict['src'] = int(lsplit[6])
    settingsdict['wavscale'] = str2bool(lsplit[7])
    settingsdict['aniso'] = float(lsplit[8])
    settingsdict['freqbase'] = float(lsplit[9])

    lsplit = lines[9].strip().split()
    settingsdict['reduce'] = str2bool(lsplit[0])
    settingsdict['redvel'] = float(lsplit[1])
    settingsdict['tbegin'] = float(lsplit[2])
    settingsdict['fst'] = str2bool(lsplit[3])
    settingsdict['fsr'] = str2bool(lsplit[4])
    settingsdict['fsb'] = str2bool(lsplit[5])
    settingsdict['fsl'] = str2bool(lsplit[6])
    settingsdict['sponge'] = str2bool(lsplit[7])
    settingsdict['isufx'] = int(lsplit[8])

    freqs = []
    freqstart = 11
    freqend = freqstart + settingsdict['nom']/5 + 1*(not not settingsdict['nom']%5)
    [[freqs.append(float(item)) for item in line.strip().split() ] for line in lines[freqstart:freqend]]
    settingsdict['freqs'] = np.array(freqs)

    kys = []
    kystart = freqend+1
    kyend = kystart + settingsdict['nky']/5 + 1*(not not settingsdict['nky']%5)
    [[kys.append(float(item)) for item in line.strip().split() ] for line in lines[kystart:kyend]]
    settingsdict['kys'] = np.array(kys)

    lsplit = lines[kyend+1].strip().split()
    settingsdict['nslices'] = int(lsplit[0])

    slices = []
    slicestart = kyend+3
    sliceend = slicestart + settingsdict['nslices']
    for i in xrange(slicestart,sliceend):
        slices.append(lines[i].strip().split())
        slices[-1][0] = int(slices[-1][0])
        slices[-1][1] = int(slices[-1][1])
        slices[-1][2] = float(slices[-1][2])
        settingsdict['slices'] = slices

    lsplit = lines[sliceend+1].strip().split()
    settingsdict['ns'] = int(lsplit[0])
    settingsdict['isreg'] = int(lsplit[1])
    settingsdict['sspread'] = float(lsplit[2])
    settingsdict['useswt'] = str2bool(lsplit[3])

    srcs = []
    srcstart = sliceend+3
    srcend = srcstart + settingsdict['ns']
    for i in xrange(srcstart,srcend):
        srcs.append([float(item) for item in lines[i].strip().split()[1:]])
    srcs = np.array(srcs)
    settingsdict['srcs'] = srcs

    lsplit = lines[srcend+1].strip().split()
    settingsdict['nr'] = int(lsplit[0])
    settingsdict['irreg'] = int(lsplit[1])
    settingsdict['rspread'] = float(lsplit[2])
    settingsdict['userwt'] = str2bool(lsplit[3])

    recs = []
    recstart = srcend+3
    recend = recstart + settingsdict['nr']
    for i in xrange(recstart,recend):
        recs.append([float(item) for item in lines[i].strip().split()[1:]])
    recs = np.array(recs)
    settingsdict['recs'] = recs

    lsplit = lines[recend+1].strip().split()
    settingsdict['ng'] = int(lsplit[0])
    settingsdict['igreg'] = int(lsplit[1])
    settingsdict['gspread'] = float(lsplit[2])
    settingsdict['usegwt'] = str2bool(lsplit[3])

    geos = []
    geostart = recend+3
    geoend = geostart + settingsdict['ng']
    for i in xrange(geostart,geoend):
        geos.append([float(item) for item in lines[i].strip().split()[1:]])
    geos = np.array(geos)
    settingsdict['geos'] = geos

    lsplit = lines[geoend+1].strip().split()
    settingsdict['sghost'] = str2bool(lsplit[0])
    settingsdict['rghost'] = str2bool(lsplit[1])
    settingsdict['gghost'] = str2bool(lsplit[2])
    settingsdict['zgg'] = float(lsplit[3])

    lsplit = lines[geoend+3].strip().split()
    settingsdict['zero1'] = [int(item) for item in lsplit]

    lsplit = lines[geoend+4].strip().split()
    settingsdict['zero2'] = [int(item) for item in lsplit]

    return settingsdict
