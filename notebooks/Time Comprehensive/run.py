#!/usr/bin/env python

import sys
sys.path.append('/home/smithyman/Projects/zephyr')
sys.path.append('/usr/local/bin')


print('Test script for xhlayr example\n')

print('>>> ' + 'from zephyr.frontend.jobs import OmegaJob, AnisoOmegaJob')
from zephyr.frontend.jobs import OmegaJob, AnisoOmegaJob

print('>>> ' + 'j = OmegaJob(\'xhlayr\')')
j = OmegaJob('xhlayr')

print('>>> ' + 'j.run()')
j.run()
