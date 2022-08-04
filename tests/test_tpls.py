import pytest

import numpy as np
from numpy.testing import assert_allclose
from sklearn.decomposition import PCA
import tensorly as tl
from tensorly.cp_tensor import CPTensor, cp_normalize, cp_to_tensor
from tensorly.metrics.factors import congruence_coefficient
from tensorly.random import random_cp

from cmtf_pls.npls import NPLS


TENSOR_DIMENSIONS = (100, 38, 65)
N_RESPONSE = 8
N_LATENT = 8

TEST_RANKS = [3, 4, 5, 6]
TEST_RESPONSE = [1, 4, 8, 16, 32]

# Supporting Functions


def _get_pls_dataset(tensor_dimensions, n_latent, n_response):
    x_tensor = random_cp(
        tensor_dimensions,
        n_latent,
        orthogonal=True,
        normalise_factors=True,
        random_state=42
    )
    y_tensor = random_cp(
        (tensor_dimensions[0], n_response),
        n_latent,
        random_state=42
    )

    y_tensor.factors[0] = x_tensor.factors[0]
    x = cp_to_tensor(x_tensor)
    y = cp_to_tensor(y_tensor)

    return x, y, x_tensor, y_tensor


def _get_standard_synthetic():
    return _get_pls_dataset(
        TENSOR_DIMENSIONS,
        N_LATENT,
        N_RESPONSE
    )


# Class Structure Tests

@pytest.mark.parametrize('x_rank', TEST_RANKS)
@pytest.mark.parametrize('n_response', TEST_RESPONSE)
def test_transform(x_rank, n_response):
    x, y, _, _ = _get_pls_dataset(
        tuple([10] * x_rank),
        N_LATENT,
        n_response
    )
    pls = NPLS(N_LATENT)
    pls.fit(x, y)

    transformed = pls.transform(x)
    assert_allclose(transformed, pls.X_factors[0])


def test_factor_normality():
    x, y, _, _ = _get_standard_synthetic()
    pls = NPLS(n_components=N_LATENT)
    pls.fit(x, y)

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


# This method should test for factor hyper-orthogonality; components seem
# very loosely hyper-orthogonal (cut-off of 1E-2 is generous).
def test_factor_orthogonality():
    x, y, _, _ = _get_standard_synthetic()
    pls = NPLS(n_components=N_LATENT)
    pls.fit(x, y)
    x_cp = CPTensor((None, pls.X_factors))
    x_cp = cp_normalize(x_cp)

    for component_1 in range(x_cp.rank):
        for component_2 in range(component_1 + 1, x_cp.rank):
            factor_product = 1
            for factor in x_cp.factors:
                factor_product *= np.dot(
                    factor[:, component_1],
                    factor[:, component_2]
                )
            assert abs(factor_product) < 1E-8


def test_consistent_components():
    x, y, _, _ = _get_standard_synthetic()
    pls = NPLS(n_components=N_LATENT)
    pls.fit(x, y)

    for x_factor in pls.X_factors:
        assert x_factor.shape[1] == N_LATENT

    for y_factor in pls.Y_factors:
        assert y_factor.shape[1] == N_LATENT


# Dimension Compatibility Tests

@pytest.mark.parametrize('x_rank', TEST_RANKS)
@pytest.mark.parametrize('n_response', TEST_RESPONSE)
def test_dimension_compatibility(x_rank, n_response):
    x, y, _, _ = _get_pls_dataset(
        tuple([10] * x_rank),
        N_LATENT,
        n_response
    )
    try:
        pls = NPLS(N_LATENT)
        pls.fit(x, y)
    except ValueError:
        raise AssertionError(
            f'Fit failed for {len(x.shape)}-dimensional tensor with '
            f'{n_response} response variables in y'
        )


# Decomposition Accuracy Tests

def test_same_x_y_2d():
    x = random_cp(
        (100, 38),
        N_LATENT,
        orthogonal=True,
        full=True,
        random_state=42
    )
    pls = NPLS(N_LATENT)
    pca = PCA(N_LATENT)

    pls.fit(x, x)
    scores = pca.fit_transform(x)

    assert_allclose(pls.X_factors[0], pls.Y_factors[0], rtol=0, atol=1E-4)
    assert_allclose(pls.X_factors[1], pls.Y_factors[1], rtol=0, atol=1E-4)
    assert congruence_coefficient(pls.X_factors[0], scores)[0] > 0.95
    assert congruence_coefficient(pls.X_factors[1], pca.components_.T)[0] > 0.95


def test_zero_covariance_x():
    x, y, _, _ = _get_standard_synthetic()
    x[:, 0, :] = 1
    pls = NPLS(N_LATENT)
    pls.fit(x, y)

    assert_allclose(
        pls.X_factors[1][0, :],
        0
    )


def test_zero_covariance_y():
    x, y, _, _ = _get_standard_synthetic()
    y[:, 0] = 1
    pls = NPLS(N_LATENT)
    pls.fit(x, y)

    assert_allclose(
        pls.Y_factors[1][0, :],
        0
    )


@pytest.mark.parametrize('x_rank', TEST_RANKS)
@pytest.mark.parametrize('n_response', TEST_RESPONSE)
def test_decomposition_accuracy(x_rank, n_response):
    x, y, x_cp, y_cp = _get_pls_dataset(
        tuple([10] * x_rank),
        N_LATENT,
        n_response
    )
    pls = NPLS(N_LATENT)
    pls.fit(x, y)

    cp_normalize(x_cp)

    for pls_factor, true_factor in zip(pls.X_factors, x_cp.factors):
        assert congruence_coefficient(pls_factor, true_factor)[0] > 0.85

    assert congruence_coefficient(pls.Y_factors[1], y_cp.factors[1])[0] > 0.85


def test_reconstruction_x():
    x, y, _, _ = _get_standard_synthetic()
    pls = NPLS(N_LATENT)
    pls.fit(x, y)

    assert_allclose(pls.X_reconstructed(), x)


# def test_reconstruction_y():
#     x, y, _, _ = _get_standard_synthetic()
#     pls = NPLS(N_LATENT)
#     pls.fit(x, y)
#
#     assert_allclose(pls.Y_reconstructed(), y)
