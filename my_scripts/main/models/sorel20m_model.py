from ember import PEFeatureExtractor
from .abstract_model import AbstractModel
import numpy as np
from numpy import ndarray

class SOREL20M_Model(AbstractModel):
    def __init__(self):
        super().__init__()
        self._extractor = PEFeatureExtractor(feature_version=2)
    
    @property
    def extractor(self):
        return self._extractor

    def do_extract_features(self, raw_bytes):
        raw_features = self.extractor.raw_features(raw_bytes)
        feature_vector = self.extractor.process_raw_features(raw_features)

        # Apply SOREL-20M log postprocessing to extracted features.
        # NOTE: This is ChatGPT-generated code, and I haven't found
        # any official documentation of SOREL-20M that mentions this
        # procedure in its feature engineering phase. Therefore it is
        # generally recommended that you do not uncomment the code
        # below.
        #
        # return self.sorel_postproc(feature_vector)

        return feature_vector
    
    @staticmethod
    def sorel_postproc(x):
        x = np.asarray(x, dtype=np.float32)
        lz = x < 0
        gz = x > 0
        x[lz] = -np.log(1 - x[lz])
        x[gz] = np.log(1 + x[gz])
        return x
