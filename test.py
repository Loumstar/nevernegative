import time
from pathlib import Path

import imageio
import numpy as np
import rawpy
from wand.image import Image
from wand.resource import limits

path = Path(
    "/Users/louismanestar/Library/CloudStorage/GoogleDrive-louis.manestar@gmail.com/My Drive/Film Photos/25-03-06 PX:Home:Tate Nikon P3200/scans/v1/DSC_2436.NEF"
)

print(
    limits.get_resource_limit("memory"),
    limits.get_resource_limit("thread"),
    limits.get_resource_limit("throttle"),
)


start = time.time()
with Image(filename=path.as_posix(), format="raw") as image:
    numpy_image = np.array(image)

delta = time.time() - start
print(f"Wand: {delta:.3f}s")

start = time.time()
with Image(filename=path.as_posix()) as image:
    numpy_image = np.array(image)

delta = time.time() - start
print(f"Wand (no format): {delta:.3f}s")

start = time.time()
with rawpy.imread(path.as_posix()) as image:
    numpy_image = image.postprocess()

delta = time.time() - start
print(f"RawPy: {delta:.3f}s")

start = time.time()
im = imageio.v3.imread(path)
delta = time.time() - start
print(f"ImageIO: {delta:.3f}s")

# start = time.time()
# with Raw(filename=path.as_posix()) as raw:
#     raw.options.white_balance = WhiteBalance(camera=False, auto=False)
#     numpy_image = raw.process()

# delta = time.time() - start
# print(f"RawKit: {delta:.3f}s")
