import matplotlib.pyplot as plt
import numpy as np

from cmtf_pls.figures.common import getSetup
from cmtf_pls.pls3 import ThreeModePLS
from cmtf_pls.synthetic import import_synthetic
from cmtf_pls.validate import get_q2y

TENSOR_DIMENSIONS = (100, 38, 65)
N_RESPONSE = 1
N_LATENT = 8


def makeFigure():
    x, y, _ = import_synthetic(
        TENSOR_DIMENSIONS,
        N_RESPONSE,
        N_LATENT
    )
    components = np.arange(1, 11, 1)
    q2ys = np.zeros(components.shape)

    for index, n_components in enumerate(components):
        pls_tensor = ThreeModePLS(n_components)
        pls_tensor.fit(x, y)
        q2ys[index] = get_q2y(pls_tensor)

    axs, fig = getSetup(
        (4, 3),
        {
            'ncols': 1,
            'nrows': 1
        }
    )
    ax = axs[0]

    ax.set_xticks(components)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Q2Y')

    ax.plot(components, q2ys)

    return fig
