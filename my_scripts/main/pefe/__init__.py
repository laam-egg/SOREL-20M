from pefe_agent.consumer import PEFEConsumer
import hashlib
from ..models import SOREL20M_Model
from numpy import ndarray

_model = SOREL20M_Model()
def inspect(pe_file_path):
    # type: (str) -> tuple[str, ndarray]
    with open(pe_file_path, "rb") as f:
        raw_bytes = f.read()
    
    id = hashlib.sha256(raw_bytes).digest()
    X = _model.extract_features(raw_bytes)

    return id, X

class SOREL20M_PEFEConsumer(PEFEConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle_pe_file(self, path):
        return inspect(path)

def run_pefe():
    SOREL20M_PEFEConsumer("SOREL20M_PEFE").run()
