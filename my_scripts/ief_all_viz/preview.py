from pathlib import Path
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

def load_results(results_dir):
    # type: (str) -> None
    INDEX_FILE_PATH = Path(results_dir) / "index.msgpack"
    with open(INDEX_FILE_PATH, 'rb') as index_file:
        RESULTS = msgpack.unpack(index_file, raw=False)
    return RESULTS

def main():
    import sys
    results_dir = sys.argv[1]
    RESULTS = load_results(results_dir)
    from pprint import pprint
    pprint(RESULTS)

if __name__ == "__main__":
    main()
