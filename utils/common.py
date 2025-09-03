import numpy as np

__all__ = ['to_categorical']

def to_categorical(target: np.ndarray, n_classes: int = None) -> np.ndarray:
	n_classes = n_classes if n_classes is not None else np.max(target) + 1
	batch_size = target.shape[0]
	one_hot = np.zeros((batch_size, n_classes))
	one_hot[np.arange(batch_size), target] = 1
	return one_hot
