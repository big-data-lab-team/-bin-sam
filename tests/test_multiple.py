import pytest
import numpy as np
import subprocess as sp
import hashlib
import test_helpers as th
from os import path as op, makedirs
from sam import imageutils as iu
import nibabel as nib

strategy = 'multiple'
data_folder = th.get_data_folder()
out_folder = th.get_out_folder(strategy)
exp_recon = op.join(data_folder, "test_reconstructed.nii")
mem = 1024**3

try:
    makedirs(out_folder)
except Exception as e:
    pass


def test_multiple_writes():

    im = iu.ImageUtils(filepath=exp_recon)

    im.split_multiple_writes(Y_splits=2, Z_splits=2, X_splits=2,
                             out_dir=out_folder, mem=mem,
                             filename_prefix=strategy, extension="nii")

    expected_filenames = th.get_list_data()
    out_filenames = th.get_list_out(strategy)

    for i in range(0, len(expected_filenames)):
        with open(expected_filenames[i], 'rb') as expected_data:
            with open(out_filenames[i], 'rb') as observed_data:
                expected_hash = hashlib.md5(expected_data.read()).hexdigest()
                observed_hash = hashlib.md5(observed_data.read()).hexdigest()

                print(expected_hash, observed_hash)
                assert expected_hash == observed_hash


def test_multiple_reads():

    out_recon = op.join(out_folder, "m_reconstructed.nii")

    im = iu.ImageUtils(filepath=out_recon,
                       first_dim=20, second_dim=20, third_dim=20,
                       dtype=np.ushort)
    im.merge(op.join(out_folder, 'legend.txt'), strategy,
                       mem=mem)

    img1=nib.load(exp_recon).get_fdata()
    img2=nib.load(out_recon).get_fdata()
    assert img1.shape==img2.shape
    assert (img1==img2).all()

    '''with open(exp_recon, 'rb') as exp_data:
        with open(out_recon, 'rb') as out_data:
            expected_hash = hashlib.md5(exp_data.read()).hexdigest()
            observed_hash = hashlib.md5(out_data.read()).hexdigest()

            print(expected_hash, observed_hash)
            assert expected_hash == observed_hash'''
