#!/usr/bin/env python

import sys
sys.path.append('/home/smithyman/Projects/zephyr')
sys.path.append('/usr/local/bin')

from zephyr.frontend.jobs import OmegaJob, AnisoOmegaJob

j = OmegaJob('xhlayr')
j.run()
