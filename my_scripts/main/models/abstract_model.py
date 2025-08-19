import numpy as np
from numpy import ndarray
import time

class AbstractModel:
    def __init__(self):
        pass

    def load(self, model_path):
        # type: (AbstractModel, str) -> None
        raise NotImplementedError
    
    def extract_features(self, pe_file_path):
        # type: (AbstractModel, str) -> ndarray
        raise NotImplementedError
    
    def predict(self, feature_vectors):
        # type: (AbstractModel, ndarray) -> None
        raise NotImplementedError
    
    def predict_single_file(self, pe_file_path):
        # type: (AbstractModel, str) -> float

        extract_features_start_time = time.perf_counter()
        feature_vector = self.extract_features(pe_file_path)
        extract_features_end_time = time.perf_counter()
        extract_features_time = extract_features_end_time - extract_features_start_time

        feature_vectors = np.array([feature_vector])

        inference_start_time = time.perf_counter()
        prob = self.predict(feature_vectors)[0]
        inference_end_time = time.perf_counter()
        inference_time = inference_end_time - inference_start_time

        print(f"... feature extraction took {extract_features_time:.6f}s ; inference {inference_time:.6f}s ...")

        return prob
