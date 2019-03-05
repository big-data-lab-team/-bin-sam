from os import path as op
import nibabel as nib

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

def assert_img_content(exp_recon, out_recon):
    img1=nib.load(exp_recon)
    img1_data=img1.get_fdata()
    img2=nib.load(out_recon)
    img2_data=img2.get_fdata()
    assert img1_data.shape==img2_data.shape
    assert (img1_data==img2_data).all()
    assert img1.header==img2.header
