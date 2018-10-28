import pytest
import numpy as np
import subprocess as sp
import hashlib
from . import test_helpers as th
from os import path as op, makedirs
from .. import imageutils as iu


strategy = 'clustered'
exp_fldr = th.get_exp_fldr()
out_fldr = th.get_out_fldr(strategy)
exp_recon = op.join(exp_fldr, "test_reconstructed.nii")
mem=1024**3

try:
    makedirs(out_fldr)
except Exception as e:
    pass


def test_clustered_writes():

    exp_fldr = th.get_exp_fldr()
    out_fldr = th.get_out_fldr(strategy)

    im = iu.ImageUtils(filepath=exp_recon)

    im.split_clustered_writes(Y_splits=2, Z_splits=2, X_splits=2,
                              out_dir=out_fldr, mem=mem, 
                              filename_prefix=strategy, extension="nii")

    expected_filenames = th.get_list_exp()
    out_filenames = th.get_list_out(strategy)

    for i in range(0, len(expected_filenames)):
        with open(expected_filenames[i], 'rb') as expected_data:
            with open(out_filenames[i], 'rb') as observed_data:
                expected_hash = hashlib.md5(expected_data.read()).hexdigest()
                observed_hash = hashlib.md5(observed_data.read()).hexdigest()

                print(expected_hash, observed_hash)
                assert expected_hash == observed_hash

def test_clustered_reads():                                                       
                                                                                 
    out_recon = op.join(out_fldr, "c_reconstructed.nii")                         
                                                                                 
    im = iu.ImageUtils(filepath=out_recon,                                       
                        first_dim=20, second_dim=20, third_dim=20,               
                        dtype=np.ushort)                                         
    im.reconstruct_img(op.join(out_fldr, 'legend.txt'), strategy,                
                       mem=mem)

    compare_recon_files(exp_recon, out_recon)

def test_clustered_reads_nomem():

    out_recon = op.join(out_fldr, "c_reconstructed_0.nii")                         
                                                                                 
    im = iu.ImageUtils(filepath=out_recon,                                       
                        first_dim=20, second_dim=20, third_dim=20,               
                        dtype=np.ushort)                                         
    im.reconstruct_img(op.join(out_fldr, 'legend.txt'), strategy,                
                       mem=0)

    compare_recon_files(exp_recon, out_recon)

def compare_recon_files(exp_recon, out_recon):
    with open(exp_recon, 'rb') as exp_data:                                      
        with open(out_recon, 'rb') as out_data:                                  
            expected_hash = hashlib.md5(exp_data.read()).hexdigest()             
            observed_hash = hashlib.md5(out_data.read()).hexdigest()             
                                                                                 
            print(expected_hash, observed_hash)                                  
            assert expected_hash == observed_hash
