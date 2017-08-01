# -bin-sam
A library to split and merge high-resolution 3D images.
Currently supports [nii](https://nifti.nimh.nih.gov) format only.

# prerequisites
- Python 2.7.5
- Numpy 1.12.1
- Nibabel


# how to use
### Split:
#### naive split strategy
```python
import imageutils as iu
img = iu.ImageUtils(filepath="/path/to/image.nii")
img.split(first_dim=5, second_dim=5, third_dim=5, local_dir="/path/to/output_dir", filename_prefix="bigbrain")
```


#### multiple writes strategy

```python
# mem = 12g
import imageutils as iu
img = iu.ImageUtils(filepath="/path/to/image.nii")
img.split_multiple_writes(Y_splits=5, Z_splits=5, X_splits=5, out_dir="/path/to/output_dir", mem=12*1024**3, filename_prefix="bigbrain",
                          extension="nii")
```

### Merge:

#### buffered slices strategy:

```python
import imageutils as iu
import numpy as np
img = iu.ImageUtils(reconstructed="/path/to/img.nii", first_dim=3850, second_dim=3025, third_dim=3500, dtype=np.uint16)
img.reconstruct_img(legend="/path/to/legend", "clustered", mem=0)
```


#### buffered blocks strategy:

- cluster reads strategy
```python
# mem=12g
import imageutils as iu
import numpy as np
img = iu.ImageUtils(reconstructed="/path/to/img.nii", first_dim=3850, second_dim=3025, third_dim=3500, dtype=np.uint16)
img.reconstruct_img(legend="/path/to/legend", "clustered", mem=12*1024**3)
```

- multiple reads strategy
```python
# mem=12g
import imageutils as iu
import numpy as np
img = iu.ImageUtils(reconstructed="/path/to/img.nii", first_dim=3850, second_dim=3025, third_dim=3500, dtype=np.uint16)
img.reconstruct_img(legend="/path/to/legend", "multiple", mem=12*1024**3)
```


# License
<<<<<<< HEAD
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
=======
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
>>>>>>> 0c3b4339461298d32132f029b63359e748994542
