
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
@click.option('--job', default='OmegaJob', help='The job to run')
def model(projnm, job):
    '''Run a forward model'''

    from . import jobs

    jClass = getattr(jobs, job)
    assert issubclass(jClass, jobs.Job)

    j = jClass(projnm)
    j.run()
zephyr.add_command(model)

@click.command()
@click.argument('projnm')
def pack(projnm):
    '''Collect configuration into an HDF5 datafile'''

    print('Collecting project!')
    print('projnm: \t%s'%projnm)
zephyr.add_command(pack)

@click.command()
@click.argument('projnm')
def unpack(projnm):
    '''Extract configuration from an HDF5 datafile'''

    print('Extracting project!')
    print('projnm: \t%s'%projnm)
zephyr.add_command(unpack)


if __name__ == "__main__":
    zephyr()
