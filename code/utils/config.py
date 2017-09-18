import tensorflow as tf

import glob

try:
    with open('config.json') as f:
        config = eval(f.read())
except SyntaxError:
    print('Uh oh... I could not parse the config file. Is it typed correctly? --- utils.config ')
except IOError:
    print('Uh oh... I could not find the config file. --- utils.config')

def is_file_prefix(attr):
    ''' Determine whether or not the filename under field `attr` in config.json
        is a prefix of any actual file. For instance, this comes in handy when
        seeing whether or not we have a model checkpoint saved.
    '''
    return bool(glob.glob(get(attr) + '*'))

def get(attr, root=config):
    ''' Return value of specified configuration attribute. '''
    node = root
    for part in attr.split('.'):
        node = node[part]
    return node
