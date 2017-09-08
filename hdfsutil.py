from hdfs import InsecureClient
from urlparse import urlparse



class HDFSUtil():
    def __init__(self, filepath, name_node_url, user):
        self.client = InsecureClient(name_node_url, user=user)
        self.filepath = self._parse_filepath(filepath)

    def _parse_filepath(self, filepath):
        return urlparse(filepath).path

    # def read(self, filepath):
    #     with self.client.read('models/1.csv', delimiter='\n', encoding='utf-8') as reader:
    #         items = (line.split(',') for line in reader if line)
    #         assert dict((name, float(value)) for name, value in items) == model
#
# name_node_url = "http://localhost:50070"
# uri = "hdfs:///gao/a.nii"
# o = urlparse("/a/b.ss")
#
# scheme = o.scheme
# path = o.path
#
# print o
#

