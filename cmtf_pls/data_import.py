import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao


def make_synthetic_test(factors: list, test_samples: int, error: float = 0,
                        seed: int = 215):
    """
    Generates test set from given factors.

    Parameters:
        factors (list[np.array]): CP factors; first element is assumed to be
            subject factors
        test_samples (int): samples in testing set
        error (float, default: 0): standard error of added gaussian noise
        seed (int, default: 215): seed for random number generator
    """
    rng = np.random.default_rng(seed)
    dimensions = (test_samples, *(matrix.shape[0] for matrix in factors[1:]))

    factors_kr = khatri_rao(factors, skip_matrix=0)
    test_factor = rng.normal(
        0,
        1,
        size=(test_samples, factors[0].shape[1])
    )

    tensor = tl.dot(test_factor, factors_kr.T)
    tensor = tl.fold(tensor, 0, dimensions)
    tensor += rng.normal(0, error, size=dimensions)

    return tensor, test_factor


def import_synthetic(train_dimensions: tuple, n_response: int, n_latent: int,
                     error: float = 0, seed: int = 215):
    """
    Generates synthetic data.

    Parameters:
        train_dimensions (tuple): dimensions of training data
        n_response (int): number of response variables
        n_latent (int): number of latent variables in synthetic data
        error (float, default: 0): standard error of added gaussian noise
        seed (int, default: 215): seed for random number generator
    """
    rng = np.random.default_rng(seed)

    x_factors = [
        rng.normal(
            0,
            1,
            size=(train_dimensions[0], n_latent)
        )
    ]
    y_factors = [
        rng.normal(
            0,
            1,
            size=(train_dimensions[0], n_latent)
        ),
        rng.normal(
            0,
            1,
            size=(n_response, n_latent)
        )
    ]

    for dimension in train_dimensions[1:]:
        x_factors.append(
            rng.normal(
                0,
                1,
                size=(dimension, n_latent)
            )
        )

    factors_kr = khatri_rao(x_factors, skip_matrix=0)
    x = tl.dot(x_factors[0], factors_kr.T)
    x = tl.fold(x, 0, train_dimensions)
    x += rng.normal(0, error, size=train_dimensions)

    y = tl.dot(y_factors[0], y_factors[1].T)
    y += rng.normal(0, error, size=(train_dimensions[0], n_response))

    return x, y, x_factors, y_factors
