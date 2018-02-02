
import argparse

class Options(object):
    pass
GlobalOpts = Options()

def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def ParseArgs(description, additionalArgs):
    parser = argparse.ArgumentParser(description=description)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--gpuMemory', help='A float between 0 and 1. The fraction of available memory to use.', action='store', type=restricted_float, dest='gpuMemory', required=True)
    required.add_argument('--numSteps', help='The number of steps to train for.', action='store', type=int, dest='numSteps', required=True)
    for argDict in additionalArgs:
        required.add_argument(argDict['flag'],
                              help=argDict['help'],
                              action=argDict['action'],
                              type=argDict['type'],
                              dest=argDict['dest'],
                              required=argDict['required'])

    if data:
        required.add_argument('--data', help='The data set to use. One of X, Y, Z, XYZ, 3D.', action='store', dest='data', required=True)
    args = parser.parse_args()
    if data:
        GlobalOpts.data = args.data
    GlobalOpts.gpuMemory = args.gpuMemory
    GlobalOpts.numSteps = args.numSteps
