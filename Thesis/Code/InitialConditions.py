import jax.numpy as jnp
import numpy as np
import jax.random as random
import os
from utils import get_random_points_on_sphere


def continue_IC(path : str, *args):
    cells = np.load(path)[-1].reshape(-1, 3,3)

    if os.path.isfile(path[:-4] + '_properties.npy'):
        properties = np.load(path[:-4] + '_properties.npy')
    else:
        properties = np.zeros(cells.shape[0])


    return cells, properties



def egg_half_IC(N : int) -> jnp.ndarray:
    pos = get_random_points_on_sphere(N, 42)

    p = pos #/ jnp.linalg.norm(pos, axis=1, keepdims=True)

    pos = pos * G["egg_shape"]

    q = get_random_points_on_sphere(N, 43)

    cells = jnp.stack([pos, p, q], axis=1)

    cell_properties = jnp.where(pos[:,2] < 0, 0, 1)
    # cell_properties = jnp.ones(N)

    return cells, cell_properties

def egg_IC(N : int) -> jnp.ndarray:
    pos = get_random_points_on_sphere(N, 42)

    p = pos #/ jnp.linalg.norm(pos, axis=1, keepdims=True)

    pos = pos * G["egg_shape"]

    q = get_random_points_on_sphere(N, 43)

    cells = jnp.stack([pos, p, q], axis=1)

    cell_properties = jnp.ones(N)

    return cells, cell_properties

def plane_IC(N : int) -> jnp.ndarray:
    # pos = get_random_points_on_sphere(N, 42)
    pos = random.normal(random.PRNGKey(42), (N, 3))*100

    pos = pos.at[:,2].set(jnp.zeros(N))
    # pinting in the z direction
    p = jnp.array([0, 0, 1.0])[None, :].repeat(N, axis=0)

    q = get_random_points_on_sphere(N, 43)

    cells = jnp.stack([pos, p, q], axis=1)

    cell_properties = jnp.ones(N)


    return cells, cell_properties

