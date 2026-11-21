"""Ridge-regularised multiclass linear readout (closed form, bias unregularised)."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix

LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


class Ridge:
    def fit(self, X, y, C, lam):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        Y = np.eye(C)[y]
        self.ym = Y.mean(0)
        A = Z.T @ Z + lam * np.eye(Z.shape[1])
        self.W = np.linalg.solve(A, Z.T @ (Y - self.ym))
        return self

    def scores(self, X):
        return ((X - self.mu) / self.sd) @ self.W + self.ym

    def predict(self, X):
        return self.scores(X).argmax(1)


def fit_select(Xtr, ytr, Xva, yva, C, lambdas=LAMBDAS):
    """Choose lambda on the validation partition; return the fitted model, lambda, val acc."""
    best = None
    for lam in lambdas:
        m = Ridge().fit(Xtr, ytr, C, lam)
        a = accuracy_score(yva, m.predict(Xva))
        if best is None or a > best[2] + 1e-12:
            best = (m, lam, a)
    return best


def metrics(y, p, C):
    return dict(acc=100 * accuracy_score(y, p),
                bacc=100 * balanced_accuracy_score(y, p),
                prec=precision_score(y, p, average="macro", zero_division=0),
                rec=recall_score(y, p, average="macro", zero_division=0),
                f1=f1_score(y, p, average="macro", zero_division=0),
                cm=confusion_matrix(y, p, labels=list(range(C))).tolist())
