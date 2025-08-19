from pathlib import Path
import msgpack
import msgpack_numpy
from pprint import pprint
msgpack_numpy.patch()

def ief_all_viz(results_dir):
    # type: (str) -> None
    INDEX_FILE_PATH = Path(results_dir) / "index.msgpack"
    with open(INDEX_FILE_PATH, 'rb') as index_file:
        RESULTS = msgpack.unpack(index_file, raw=False)
    pprint(RESULTS)
