
import argparse

class Options(object):
    pass
GlobalOpts = Options()

def ParseArgs(description, data=False):
    parser = argparse.ArgumentParser(description=description)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--summaryDir', help='The directory to save summaries in.', action='store', dest='summaryDir', required=True)
    required.add_argument('--checkpointDir', help='The directory to save checkpoints in.', action='store', dest='checkpointDir', required=True)
    required.add_argument('--gpuMemory', help='A float between 0 and 1. The fraction of available memory to use.', action='store', dest='gpuMemory', required=True)
    required.add_argument('--numSteps', help='The number of steps to train for.', action='store', dest='numSteps', required=True)
    if data:
        required.add_argument('--data', help='The data set to use. One of X, Y, Z, XYZ, 3D.', action='store', dest='data', required=True)
    args = parser.parse_args()
    GlobalOpts.data = args.data
    GlobalOpts.summaryDir = args.summaryDir
    GlobalOpts.checkpointDir = args.checkpointDir
    GlobalOpts.gpuMemory = args.gpuMemory
    GlobalOpts.numSteps = args.numSteps
