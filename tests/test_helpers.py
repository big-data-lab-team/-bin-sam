from os import path as op


def get_data_folder():
    return 'tests/test_data/'


def get_out_folder(strategy):
    return op.join('tests/test_outputs/', strategy)


def get_list_data():
    with open(op.join(get_data_folder(), 'legend.txt'), 'r') as legend:
        return legend.read().split()


def get_list_out(strategy):
    with open(op.join(get_out_folder(strategy), 'legend.txt'), 'r') as legend:
        return legend.read().split()
