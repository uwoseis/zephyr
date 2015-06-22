#!/bin/bash

export PROFILE="mpi"
export CONFDIR="/home/travis/.ipython/profile_${PROFILE}"

ipython profile create --parallel --profile $PROFILE

cat > $CONFDIR/ipcluster_config.py << END
c = get_config()
c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'
END

cat > $CONFDIR/ipcontroller_config.py << END
c = get_config()
c.HubFactory.ip = '127.0.0.1'
END

cat > $CONFDIR/ipengine_config.py << END
c = get_config()
c.MPI.use = 'mpi4py'
END
