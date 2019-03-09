from .base import BaseUpdateablePolicy


class SoftmaxPolicy(BaseUpdateablePolicy):
    def update(self, X, A, advantages):
        # TODO: implement weighted replication etc.
        raise NotImplementedError('update')

    def batch_eval(self, X_s):
        return self.classifier.predict_proba(X_s)
