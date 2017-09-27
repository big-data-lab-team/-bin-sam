#!/usr/bin/env python
# Ref: imageutils.py from
# https://github.com/big-data-lab-team/sam/blob/master/imageutils.py
import imageutils as img_utils
from time import time
import argparse
import random
import os
from hdfs import InsecureClient

# example
# 3221225472 9663676416 6442450944 13043807040
# python sam-split.py -f cwrites -m 3221225472 9663676416 6442450944 13043807040 -r 1 -d hdfs -u http://consider.encs.concordia.ca:50070
#python sam-split.py -f mwrites -m 6442450944 -r 1 -d hdfs -u http://hdfs-cluster-h5dl:50070
# python sam-split.py -f mwrites -m 13043807040 -r 1 -d hdd



data_file = "/data/gao/data.dat"
# HDD:
ori_image_hdd = "/data/bigbrain_40microns.nii"
out_dir_hdd = "/data/gao/blocks125"

# SSD:
ori_image_ssd = "/home/gao/bigbrain_40microns.nii"
out_dir_ssd = "/home/gao/blocks125"

# HDFS:
ori_image_hdfs="/data/bigbrain_40microns.nii"
out_dir_hdfs="/data/gao/blocks125"

Y_splits=5
Z_splits=5
X_splits=5

files = {
    "hdd": (ori_image_hdd, out_dir_hdd),
    "ssd": (ori_image_ssd, out_dir_ssd),
    "hdfs": (ori_image_hdfs, out_dir_hdfs)
}

def benchmark_cwrites(mem, ori_image, out_dir, client):
    # img = img_utils.ImageUtils(ori_image, utils=hdfsutil.HDFSUtil(ori_image, name_node_url=name_node_url, user=user))
    img = img_utils.ImageUtils(ori_image)
    s_time = time()
    total_read_time, total_write_time, total_seek_time, total_seek_number = img.split_clustered_writes(Y_splits, Z_splits, X_splits, out_dir, mem, filename_prefix="bigbrain",
                          extension="nii",hdfs_client= client)
    total_time = time() - s_time
    print "mem = {}, takes {}".format(mem, total_time)

    print total_read_time, total_write_time, total_seek_time, total_seek_number, total_time
    return (total_read_time, total_write_time, total_seek_time, total_seek_number, total_time)

def benchmark_mwrites(mem, ori_image, out_dir, client):
    # img = img_utils.ImageUtils(ori_image, utils=hdfsutil.HDFSUtil(ori_image, name_node_url=name_node_url, user=user))
    img = img_utils.ImageUtils(ori_image)
    s_time = time()
    total_read_time, total_write_time, total_seek_time, total_seek_number = img.split_multiple_writes(Y_splits, Z_splits, X_splits, out_dir, mem, filename_prefix="bigbrain",
                          extension="nii", benchmark=True, hdfs_client=client)
    total_time = time() - s_time
    print "mem = {}, takes {}".format(mem, total_time)

    print total_read_time, total_write_time, total_seek_time, total_seek_number, total_time
    return (total_read_time, total_write_time, total_seek_time, total_seek_number, total_time)

def write_to_dat(data_dict, dat_file, func):
    with open(dat_file, "a") as f:
        f.write("# {}".format(func))
        f.write("\n")
        for k in sorted(data_dict.keys()):
            for e in data_dict[k]:
                f.write(str(e) + " ")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='SAM - split')
    parser.add_argument('-f', '--func', choices=['mwrites', 'cwrites'], help="choose split or merge function", required=True)
    parser.add_argument('-m', '--mem', nargs='+', type=int, help="mem in bytes. A list of mems is required", required=True)
    parser.add_argument('-r', '--rep', type=int, help="how many repetitions on each mem", required=True)
    parser.add_argument('-d', '--disk', choices=['ssd', 'hdd', 'hdfs'], help="running on local hdd/local ssd/hdfs", required=True)
    parser.add_argument('-u', '--url', help="url of hdfs client")
    args = parser.parse_args()

    mem_list = args.mem
    func = args.func
    rep = args.rep
    disk = args.disk
    client_url = args.url

    if disk == "hdfs" and client_url == None:
        raise Exception("Need url to setup hdfs client")

    if client_url == None:
        print "Running in local filesystem..."
        client = None
    else:
        print "Client url: {}".format(client_url)
        client = InsecureClient(client_url)

    for i in range(0, rep):
        data_dict = {}
        print "Repetition: {}".format(i)
        random.shuffle(mem_list)
        for mem in mem_list:
            os.system("echo 3 | sudo tee /proc/sys/vm/drop_caches")
            print "running {} function, mem = {}".format(func, mem)

            if disk == "hdfs":
                os.system("hdfs dfs -rm -r {}/*".format(files[disk][1]))
            else:
                os.system("rm -rf {}/*".format(files[disk][1]))


            if func == "mwrites":
                data_dict[mem] = benchmark_mwrites(mem=mem, ori_image=files[disk][0], out_dir=files[disk][1], client=client)
            elif func == "cwrites":
                data_dict[mem] = benchmark_cwrites(mem=mem, ori_image=files[disk][0], out_dir=files[disk][1], client=client)
            else:
                pass

        write_to_dat(data_dict, dat_file=data_file, func=func)

if __name__ == '__main__':
    main()
