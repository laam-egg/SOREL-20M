from .sorel20m_model import SOREL20M_Model
import lightgbm as lgb

class SOREL20M_LGBM(SOREL20M_Model):
    def load(self, model_path):
        print(f"[+] Loading LightGBM model from {model_path}...")
        self._model = lgb.Booster(model_file=model_path)
    
    def predict(self, feature_vectors):
        y_probs = self._model.predict(feature_vectors)
        return y_probs
