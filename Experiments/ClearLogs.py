"""
This file is supposed to be run up on the server, and clear out any unwanted models (checkpoints, metadata, and logs)
you can also just wipe everything...

"""
ALL = True
MODELNUMS = [0]
MODELNAME = "test"




import os
import shutil
if (os.getcwd()!="IntrinsicRewards"):
    os.chdir('..')

MODELNUMS = [str(x) for x in MODELNUMS]
fnames = os.listdir('../checkpoints/') #checkpoints are named just with the name and model
if not ALL:
    fnames = filter(lambda x: x.startswith(MODELNAME) and x[len(MODELNAME):] in MODELNUMS, fnames)

for fname in fnames:
    shutil.rmtree('checkpoints/' + fname)
    shutil.rmtree('logdir/' + fname)
    shutil.rmtree('metadata/' + fname + '.json')
