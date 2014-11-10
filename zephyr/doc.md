---
title: Organization and parallelism
author: Brendan Smithyman
date: November 10, 2014
---

## Problem organization and subproblem hierarchy

In order to carry out full-waveform inversion, one must forward model data and fields. In the particular case of the 2.5D viscoacoustic Helmholtz problem, there are several levels involved:

- **Frequency:** Each frequency results in a separate Helmholtz problem, with corresponding sources, receivers and wavenumbers. Whereas it is typical to use the same array and wavenumbers for all frequencies, this is not strictly necessary. The forward- and backward-propagated wavefields at each frequency (in the plane of the source) and the synthetic data (in the plane of each receiver) must be stored for use in the optimization algorithm.
- **Source:** Each source is, in principle, independent. The corresponding fields (forward- and backward-propagated) and data can all be treated without considering fields from other sources. However, when using direct matrix solvers (e.g., LU-factorization), there are computational benefits to solving multiple sources on the same node using pre-computed factors.
- **Wavenumber:** Each wavenumber component represents a separate forward problem. In order to benefit from solver reuse, each source should be calculated on the same node for the same wavenumber. Producing forward wavefields requires a summation over wavenumbers for each source. Producing backpropagated wavefields requires a summation over wavenumbers for each receiver, in the source plane, followed by a summation over sources. Only the final summation in this case needs to be gathered. However, the computation of data residuals must be done using the 3D receiver wavefields, and so a scatter is implied.