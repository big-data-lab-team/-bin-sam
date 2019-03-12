import pytest
import numpy as np
import subprocess as sp
import hashlib
import test_helpers as th
from os import path as op, makedirs
from sam import imageutils as iu


def test_naive_splits():

    strategy = 'naive'

    data_folder = th.get_data_folder()
    out_folder = th.get_out_folder(strategy)

    try:
        makedirs(out_folder)
    except Exception as e:
        print(e)

    im = iu.ImageUtils(filepath=op.join(data_folder,
                                        "test_reconstructed.nii"))

    im.split(first_dim=10, second_dim=10, third_dim=10,
             local_dir=out_folder, filename_prefix="naive", benchmark=True)

    expected_filenames = th.get_list_data()
    out_filenames = th.get_list_out(strategy)

    # temporary code as all splits are gzipped
    args = ['gunzip'] + out_filenames
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
    p.communicate()

    for i in range(0, len(expected_filenames)):
        with open(expected_filenames[i], 'rb') as expected_data:
            with open(out_filenames[i][:-3], 'rb') as observed_data:
                expected_hash = hashlib.md5(expected_data.read()).hexdigest()
                observed_hash = hashlib.md5(observed_data.read()).hexdigest()

                print(expected_hash, observed_hash)
                assert expected_hash == observed_hash
