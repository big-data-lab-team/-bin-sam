import pytest
import numpy as np
import subprocess as sp
import hashlib
from . import test_helpers as th
from os import path as op, makedirs
from .. import imageutils as iu


def test_naive_splits():

    strategy = 'naive'

    exp_fldr = th.get_exp_fldr()
    out_fldr = th.get_out_fldr(strategy)

    try:
        makedirs(out_fldr)
    except Exception as e:
        print(e)

    im = iu.ImageUtils(filepath=op.join(exp_fldr, 
                                        "test_reconstructed.nii"))

    im.split(first_dim=10, second_dim=10, third_dim=10,
             local_dir=out_fldr, filename_prefix="naive")

    expected_filenames = th.get_list_exp()
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
