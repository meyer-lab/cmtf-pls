from numpy.testing import assert_allclose
import tensorly as tl
from tensorly.metrics.factors import congruence_coefficient

from cmtf_pls.synthetic import import_synthetic
from cmtf_pls.tpls import NModePLS


TENSOR_DIMENSIONS = (100, 38, 65)
N_RESPONSE = 4
N_LATENT = 8


# Supporting Functions

def _get_standard_synthetic():
    x, y, cp_tensor = import_synthetic(
        TENSOR_DIMENSIONS,
        N_RESPONSE,
        N_LATENT
    )
    pls = NModePLS(N_LATENT)
    pls.fit(x, y)

    return x, y, cp_tensor, pls


# Class Structure Tests

def test_factor_normality():
    x, y, _, pls = _get_standard_synthetic()

    for x_factor in pls.X_factors[1:]:
        assert_allclose(
            tl.norm(x_factor, axis=0),
            1
        )

    for y_factor in pls.Y_factors[1:]:
        assert_allclose(
            tl.norm(y_factor, axis=0),
            1
        )


def test_consistent_components():
    x, y, _, pls = _get_standard_synthetic()

    for x_factor in pls.X_factors:
        assert x_factor.shape[1] == N_LATENT

    for y_factor in pls.Y_factors:
        assert y_factor.shape[1] == N_LATENT


# Dimension Compatibility Tests

def _test_dimension_compatibility(x_rank, n_response):
    x, y, _ = import_synthetic(
        tuple([100] * x_rank),
        n_response,
        N_LATENT
    )
    try:
        pls = NModePLS(N_LATENT)
        pls.fit(x, y)
    except ValueError:
        raise AssertionError(
            f'Fit failed for {len(x.shape)}-dimensional tensor with '
            f'{n_response} response variables in y'
        )


def test_compatibility_2d_x_1d_y():
    _test_dimension_compatibility(2, 1)


def test_compatibility_3d_x_1d_y():
    _test_dimension_compatibility(3, 1)


def test_compatibility_4d_x_1d_y():
    _test_dimension_compatibility(4, 1)


def test_compatibility_2d_x_2d_y():
    _test_dimension_compatibility(2, 4)


def test_compatibility_3d_x_2d_y():
    _test_dimension_compatibility(3, 4)


def test_compatibility_4d_x_2d_y():
    _test_dimension_compatibility(4, 4)


# Decomposition Accuracy Tests

def test_constant_y():
    x, y, _ = import_synthetic(
        TENSOR_DIMENSIONS,
        N_RESPONSE,
        N_LATENT
    )
    y[:, 0] = 1
    pls = NModePLS(N_LATENT)
    pls.fit(x, y)

    assert_allclose(
        pls.Y_factors[1][0, :],
        0
    )


def _test_decomposition_accuracy(x_rank, n_response):
    x, y, true_cp = import_synthetic(
        tuple([100] * x_rank),
        n_response,
        N_LATENT
    )
    pls = NModePLS(N_LATENT)
    pls.fit(x, y)

    for pls_factor, true_factor in zip(pls.X_factors, true_cp.factors):
        assert congruence_coefficient(pls_factor, true_factor)[0] > 0.95

    assert congruence_coefficient(pls.Y_factors[1], true_cp.y_factor)[0] > 0.95


def test_decomposition_accuracy_3d_x_1d_y():
    _test_decomposition_accuracy(3, 1)


def test_decomposition_accuracy_4d_x_1d_y():
    _test_decomposition_accuracy(4, 1)


def test_decomposition_accuracy_3d_x_2d_y():
    _test_decomposition_accuracy(3, 4)


def test_decomposition_accuracy_4d_x_2d_y():
    _test_decomposition_accuracy(4, 2)


# Reconstruction tests -- these will likely fail!

def test_reconstruction_x():
    x, y, _, pls = _get_standard_synthetic()
    assert_allclose(pls.X_reconstructed(), x)


def test_reconstruction_y():
    x, y, _, pls = _get_standard_synthetic()
    assert_allclose(pls.Y_reconstructed(), y)
