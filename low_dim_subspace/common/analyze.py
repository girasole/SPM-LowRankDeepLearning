import numpy as np
from tqdm import tqdm


def svd(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, VT = np.linalg.svd(W)
    return U, s, VT.T


def svdvals(W: np.ndarray) -> np.ndarray:
    return np.linalg.svd(W, compute_uv=False)


def compute_cosine_angle(U: np.ndarray, V: np.ndarray) -> float:
    return np.max(np.clip(svdvals(np.dot(U.T, V)), 0, 1)).item()


def compute_matrix_rank(W: np.ndarray, max_rank: int, eps: float = 1e-6) -> int:
    if np.linalg.norm(W, ord=2) < eps:
        return 0
    return min(np.linalg.matrix_rank(W).item(), max_rank)


def compute_svd_series(
    layer_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Uinit, _, Vinit = svd(layer_history[0])

    sval_series = []
    right_series = []
    left_series = []

    for W in tqdm(layer_history):
        Ut, st, Vt = svd(W)
        U, _, V = svd(np.vstack([Ut, Vt]).T @ np.vstack([Uinit, Vinit]))
        Q = U @ V.T
        Ut = Ut @ Q
        Vt = Vt @ Q

        sval_series.append(st)
        right_series.append(np.arccos(np.clip(np.sum(Vt * Vinit, axis=0), 0, 1)))
        left_series.append(np.arccos(np.clip(np.sum(Ut * Uinit, axis=0), 0, 1)))

    sval_series = np.array(sval_series)
    right_series = np.array(right_series)
    left_series = np.array(left_series)

    return sval_series, right_series, left_series
