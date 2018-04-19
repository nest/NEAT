import inspect
import importlib

def write_all_pychannels(module_names=[]):
    '''
    Rewrites the :module:`pychannels` file. The standard collection of 
    channels is included as well as all the channels in the user specified list
    of modules. A channel is defined as any class whose name contains the string
    'Channel'.

    Parameters
    ----------
    modules : list of strings
        names of module where the code looks for channels.
    '''
    # write intro pychannels file
    file = open('pychannels.py', 'w')
    file.write('\'\'\'\nThis file is automatically generated by ' \
                         + ':func:`write_all_pychannels`\n\'\'\'\n')
    file.write('import numpy as np\n\n')
    file.write('from ionchannels import SimChannel\n')
    file.close()
    # run through all modules and write the simulation channels
    module_names = ['neat.channels.channelcollection'] + module_names
    for module_name in module_names:
        module = importlib.import_module(module_name)
        channels = inspect.getmembers(module,
                                      lambda c: inspect.isclass(c) \
                                                and c.__module__ == module_name \
                                                and 'Channel' in c.__name__)
        for (channel_name, channel_class) in channels:
            print '>>> Writing simulation channel for ' + channel_name + '<<<\n'
            chan = channel_class()
            chan.write_to_py_file()

if __name__ == '__main__':
    write_all_pychannels()