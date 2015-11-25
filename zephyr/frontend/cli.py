
import click


@click.group()
@click.version_option()
def zephyr():
    '''A command-line interface for Zephyr'''
    

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
@click.option('--storage', type=click.Choice(['dir', 'hdf5']), default='dir')
@click.option('--fromini', type=click.File())
def init(projnm, storage, fromini):
    '''Set up a new modelling or inversion project'''
    
    print('Initializing project!')
    print('projnm: \t%s'%projnm)
    print('storage:\t%s'%storage)
    if fromini is not None:
        print('fromini:\t%s'%fromini.read())
zephyr.add_command(init)


@click.command()
@click.argument('projnm')
def invert(projnm):
    '''Run an inversion project'''
    
    print('Running project!')
    print('projnm: \t%s'%projnm)
zephyr.add_command(invert)


@click.command()
@click.argument('projnm')
def inspect(projnm):
    '''Print information about an existing project'''
    
    print('Information about an existing project!')
    print('projnm: \t%s'%projnm)
zephyr.add_command(inspect)


@click.command()
@click.argument('projnm')
def migrate(projnm):
    '''Run a migration'''
    
    print('Running project!')
    print('projnm: \t%s'%projnm)
zephyr.add_command(migrate)


@click.command()
@click.argument('projnm')
def model(projnm):
    '''Run a forward model'''
    
    print('Running project!')
    print('projnm: \t%s'%projnm)
zephyr.add_command(model)


if __name__ == "__main__":
    zephyr()
