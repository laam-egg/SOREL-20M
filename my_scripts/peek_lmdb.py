import lmdb, zlib, msgpack, numpy as np
from pathlib import Path
import sys
db_path = Path(sys.argv[1])

def features_postproc_func(x):
    x = np.asarray(x[0], dtype=np.float32)
    lz = x < 0
    gz = x > 0
    x[lz] = - np.log(1 - x[lz])
    x[gz] = np.log(1 + x[gz])
    return x

env = lmdb.open(str(db_path), readonly=True, lock=False, max_readers=1024)
with env.begin() as txn:
    for i in range(16):
        key, val = next(iter(txn.cursor()))
        x = msgpack.loads(zlib.decompress(val), strict_map_key=False)
        vec = features_postproc_func(x)

        print("Key:", key.decode() if isinstance(key, bytes) else key)
        print("Shape:", vec.shape)       # should be (2381,) due to using EMBER2017/2018
        print("First 10 features:", vec[:10])
