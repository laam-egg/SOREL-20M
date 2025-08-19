######################################################################
######### BEGIN SECTION COPIED FROM $PROJECT_ROOT/net.py #############
######################################################################

# Copyright 2020, Sophos Limited. All rights reserved.
# 
# 'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
# Sophos Limited and Sophos Group. All other product and company
# names mentioned are trademarks or registered trademarks of their
# respective owners.


import torch
from torch import nn
import torch.nn.functional as F

class PENetwork(nn.Module):
    """
    This is a simple network loosely based on the one used in ALOHA: Auxiliary Loss Optimization for Hypothesis Augmentation (https://arxiv.org/abs/1903.05700)

    Note that it uses fewer (and smaller) layers, as well as a single layer for all tag predictions, performance will suffer accordingly.
    """
    def __init__(self,use_malware=True,use_counts=True,use_tags=True,n_tags=None,feature_dimension=1024, layer_sizes = None):
        self.use_malware=use_malware
        self.use_counts=use_counts
        self.use_tags=use_tags
        self.n_tags = n_tags
        if self.use_tags and self.n_tags == None:
            raise ValueError("n_tags was None but we're trying to predict tags. Please include n_tags")
        super(PENetwork,self).__init__()
        p = 0.05
        layers = []
        if layer_sizes is None:layer_sizes=[512,512,128]
        for i,ls in enumerate(layer_sizes):
            if i == 0:
                layers.append(nn.Linear(feature_dimension,ls))
            else:
                layers.append(nn.Linear(layer_sizes[i-1],ls))
            layers.append(nn.LayerNorm(ls))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(p))
        self.model_base = nn.Sequential(*tuple(layers))
        if self.use_malware:
            self.malware_head = nn.Sequential(nn.Linear(layer_sizes[-1], 1),
                                            nn.Sigmoid())
        if self.use_counts:
            self.count_head = nn.Linear(layer_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()
        if self.use_tags:
            self.tag_head = nn.Sequential(nn.Linear(layer_sizes[-1],64),
                                            nn.ELU(), 
                                            nn.Linear(64,64),
                                            nn.ELU(),
                                            nn.Linear(64,n_tags),
                                            nn.Sigmoid())

    def forward(self,data):
        rv = {}
        base_result = self.model_base.forward(data)
        if self.use_malware:
            rv['malware'] = self.malware_head(base_result)
        if self.use_counts:
            rv['count'] = self.count_head(base_result)
        if self.use_tags:
            rv['tags'] = self.tag_head(base_result)
        return rv

######################################################################
########## END SECTION COPIED FROM $PROJECT_ROOT/net.py ##############
######################################################################

from .sorel20m_model import SOREL20M_Model
import torch
from copy import deepcopy
import numpy as np

class SOREL20M_FFNN(SOREL20M_Model):
    def load(self, model_path):
        print(f"[+] Loading FFNN model (checkpoint) from {model_path}...")
        model = PENetwork(
            use_malware=True,
            use_counts=False,
            use_tags=False,
            n_tags=2,
            feature_dimension=2381
        )

        state_dict = torch.load(model_path, map_location="cpu")
        # If the checkpoint was saved with a wrapper (e.g., 'model.')
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[len("model."):]
            # Skip keys that don't exist in our smaller model
            if k in model.state_dict():
                cleaned_state_dict[k] = v

        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()

        DEVICE_NAME = "cuda:0"
        try:
            model.to(DEVICE_NAME)
        except Exception as e:
            print(f"Failed to use device {DEVICE_NAME}: {e}")
            print(f"Falling back to CPU-only mode.")
            DEVICE_NAME = "cpu"

        self._DEVICE_NAME = DEVICE_NAME
        self._model = model
        print(f"Operating on device: {self._DEVICE_NAME}")
    
    def predict(self, feature_vectors):
        X = torch.Tensor(feature_vectors).to(self._DEVICE_NAME)
        raw_predictions = self._model(X)
        y_probs = self.detach_and_copy_array(raw_predictions['malware'])
        return y_probs
    
    @staticmethod
    def detach_and_copy_array(array):
        """
        Copied from $PROJECT_ROOT/evaluate.py
        """
        if isinstance(array, torch.Tensor):
            return deepcopy(array.cpu().detach().numpy()).ravel()
        elif isinstance(array, np.ndarray):
            return deepcopy(array).ravel()
        else:
            raise ValueError("Got array of unknown type {} with value {}".format(type(array), str(array)))
