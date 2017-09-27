echo "mwrites"
python sam-split.py -f mwrites -m 3221225472 9663676416 6442450944 13043807040 -r 1 -d hdfs -u http://consider.encs.concordia.ca:50070

echo "cwrites"
python sam-split.py -f cwrites -m 3221225472 9663676416 6442450944 13043807040 -r 1 -d hdfs -u http://consider.encs.concordia.ca:50070


