#!/usr/bin/env python

import click
import numpy as np

import anemoi
import windtunnel
import SimPEG

@click.group()
def zephyr():
    '''A command-line interface for Zephyr'''
    
    # Do argument parsing here (eg. with argparse) and anything else
    # you want your project to do.

@click.command()
@click.argument('projnm')
@click.confirmation_option(prompt='Are you sure you want to clean project outputs?')
def clean(projnm):
    '''Clean up project results / outputs'''
    
    print('Cleaning up project!')
    print('projnm: \t%s'%projnm)
zephyr.add_command(clean)
    
@click.command()
@click.argument('projnm')
@click.option('--storage', '-s', type=click.Choice(['dir', 'hdf5']), default='dir')
def init(projnm, storage):
    '''Set up a new modelling or inversion project'''
    
    print('Initializing project!')
    print('projnm: \t%s'%projnm)
    print('storage:\t%s'%storage)
zephyr.add_command(init)

@click.command()
@click.argument('projnm')
def inspect(projnm):
    '''Print information about an existing project'''
    
    print('Information about an existing project!')
    print('projnm: \t%s'%projnm)
zephyr.add_command(inspect)
    
@click.command()
@click.argument('projnm')
def run(projnm):
    '''Run a modelling or inversion project'''
    
    print('Running project!')
    print('projnm: \t%s'%projnm)
zephyr.add_command(run)


if __name__ == "__main__":
    zephyr()
