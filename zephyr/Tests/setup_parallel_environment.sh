#!/bin/bash

export PROFILE="mpi"

ipython profile create --parallel --profile $PROFILE

cat > $HOME/.ipython/profile_${PROFILE}/ipcluster_config.py << END
c = get_config()
c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'
END

cat > $HOME/.ipython/profile_${PROFILE}/ipcontroller_config.py << END
c = get_config()
c.HubFactory.ip = '127.0.0.1'
END

cat > $HOME/.ipython/profile_${PROFILE}/ipengine_config.py << END
c = get_config()
c.MPI.use = 'mpi4py'
END
