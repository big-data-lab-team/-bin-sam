from os import path as op


def get_exp_fldr():
    return 'tests/test_data/'

def get_out_fldr(strategy):
    return op.join('tests/test_outputs/', strategy)

def get_list_exp():
    with open(op.join(get_exp_fldr(), 'legend.txt'), 'r') as legend:
        return legend.read().split()

def get_list_out(strategy):
    with open(op.join(get_out_fldr(strategy), 'legend.txt'), 'r') as legend:
        return legend.read().split()
