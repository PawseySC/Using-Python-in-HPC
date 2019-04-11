#!/usr/bin/env python

from mpi4py import MPI
import numpy

# Function to calcualte pi that each MPI rank will use
def compute_pi(samples):
    count = 0
    for x, y in samples:
        if x**2 + y**2 <= 1:
            count += 1
    pi = 4*float(count)/len(samples)
    return pi

# Set up our MPI environment
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

# Processor 0 generates random samples that each processor will use
if myrank == 0:
    N = 10000000 // nprocs
    samples = numpy.random.random((nprocs, N, 2))
else:
    samples = None

# Distribute the samples amongst all processors wiht MPI_Scatter
samples = comm.scatter(samples, root=0)

# Each processors calculates their value of pi (we'll take the average)
mypi = compute_pi(samples) / nprocs

# MPI_Reduce collects all individual
pi = comm.reduce(mypi, op=MPI.SUM, root=0)

if myrank == 0:
    error = abs(pi - numpy.pi)
    print("pi is approximately %.16f, error is %.16f" % (pi, error))
