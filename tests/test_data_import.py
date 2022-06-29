import numpy as np

from cmtf_pls.data_import import import_synthetic, make_synthetic_test

TENSOR_DIMENSIONS = (100, 38, 65)
N_RESPONSE = 4
N_LATENT = 8


def test_synthetic_dimensions():
    x, y, x_factors, y_factors = import_synthetic(
        TENSOR_DIMENSIONS,
        N_RESPONSE,
        N_LATENT,
        error=0
    )

    assert all([factor.shape[1] == N_LATENT for factor in x_factors])
    assert all([factor.shape[1] == N_LATENT for factor in y_factors])
    assert x.shape == TENSOR_DIMENSIONS
    assert y.shape == (TENSOR_DIMENSIONS[0], N_RESPONSE)


def test_synthetic_test_dimensions():
    x, y, x_factors, y_factors = import_synthetic(
        TENSOR_DIMENSIONS,
        N_RESPONSE,
        N_LATENT,
        error=0
    )
    x_test, x_test_factors = make_synthetic_test(x_factors, 10, 0)

    assert x_factors[0].shape[1] == x_test_factors.shape[1]


def test_reproducibility():
    x1, y1, x_factors1, y_factors1 = import_synthetic(
        TENSOR_DIMENSIONS,
        N_RESPONSE,
        N_LATENT,
        error=0,
        seed=42
    )
    x2, y2, x_factors2, y_factors2 = import_synthetic(
        TENSOR_DIMENSIONS,
        N_RESPONSE,
        N_LATENT,
        error=0,
        seed=42
    )
    x3, y3, x_factors3, y_factors3 = import_synthetic(
        TENSOR_DIMENSIONS,
        N_RESPONSE,
        N_LATENT,
        error=0,
        seed=43
    )

    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert not np.array_equal(x1, x3)
    assert not np.array_equal(y1, y3)
