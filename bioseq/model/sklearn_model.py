from sklearn import linear_model, ensemble

from .base_model import BaseModel
from bioseq.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register('linear')
class LinearRegressionModel(BaseModel):

    _name = "linear"

    def __init__(self) -> None:
        super().__init__()
        self._model = linear_model.LinearRegression()

    def train(self, X, labels):
        flattened = X.reshape(X.shape[0], -1)
        self._model.fit(flattened, labels)

    def reset(self):
        self._model = linear_model.LinearRegression()
        super().reset()

    def _fit(self, X):
        flattened = X.reshape(X.shape[0], -1)
        return self._model.predict(flattened)


@MODEL_REGISTRY.register('random_forest')
class RandomForestModel(BaseModel):

    _name = "random_forest"

    def __init__(self) -> None:
        super().__init__()
        self._model = ensemble.RandomForestRegressor()

    def train(self, X, labels):
        flattened = X.reshape(X.shape[0], -1)
        self._model.fit(flattened, labels)

    def reset(self):
        self._model = ensemble.RandomForestRegressor()
        super().reset()

    def _fit(self, X):
        flattened = X.reshape(X.shape[0], -1)
        return self._model.predict(flattened)
