import nibabel as nib
from hdfs import InsecureClient

from io import BytesIO
# /user/gao/
client = InsecureClient("http://localhost:50070", user="gao")

# file = "/Users/amazin/WorkStudio/data/bigbrain_1540_1815_0.nii"

file = "/data/bigbrain_1540_1815_0.nii"



with client.read(file, length=352) as header:

    fh = nib.FileHolder(fileobj= BytesIO(header.read()))
    img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})

    print img.header


  #
# with client.read('models/1.csv', delimiter='\n', encoding='utf-8') as reader:
#   items = (line.split(',') for line in reader if line)
#   assert dict((name, float(value)) for name, value in items) == model