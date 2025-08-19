from .sorel20m_model import SOREL20M_Model
import lightgbm as lgb

class SOREL20M_LGBM(SOREL20M_Model):
    def do_load(self, model_path):
        self._model = lgb.Booster(model_file=model_path)
    
    def do_predict(self, feature_vectors):
        y_probs = self._model.predict(feature_vectors)
        return y_probs
    
    def do_get_batch_size(self):
        return 65536
