import jax.numpy as jnp
import numpy as np
import jax.random as random
import os
from utils import get_random_points_on_sphere

# from G import G

# egg_shape = G["egg_shape"]
sphere_shape = jnp.array([20, 20, 20])
egg_shape = jnp.array([41.5, 41.5/3, 41.5/3])


# egg_shape = jnp.array([230, 230/3, 230/3])


def continue_IC(path : str, *args):
    cells = np.load(path)[-1].reshape(-1, 3,3)

    if os.path.isfile(path[:-4] + '_properties.npy'):
        properties = np.load(path[:-4] + '_properties.npy')
    else:
        properties = np.zeros(cells.shape[0])


    return cells, properties

def make_better_egg_pos(cells):
    cells = cells.at[2].set(cells[2] + jnp.square(jnp.abs(cells[0]/egg_shape[0]))*egg_shape[2]/2)
    return cells

def inv_make_better_egg_pos(cells):
    cells = cells.at[2].set(cells[2] - jnp.square(jnp.abs(cells[0]/egg_shape[0]))*egg_shape[2]/2)
    return cells

def make_better_egg(cells):
    cells = cells.at[:,0,2].set(cells[:,0,2] + jnp.square(jnp.abs(cells[:,0,0]/egg_shape[0]))*egg_shape[2]/2)
    return cells

def egg_IC(N : int) -> jnp.ndarray:
    pos = get_random_points_on_sphere(N, 42)

    p = pos #/ jnp.linalg.norm(pos, axis=1, keepdims=True)

    # make them all point in the same direction, whirl around the x axis
    q = pos.copy()
    q = q.at[:,0].set(0)
    q = q / jnp.linalg.norm(q, axis=1, keepdims=True)
    _y = q[:,1]
    _z = q[:,2]
    q = q.at[:,1].set(-_z)
    q = q.at[:,2].set(_y)

    pos = pos * egg_shape

    cells = jnp.stack([pos, p, q], axis=1)

    cell_properties = jnp.ones(N)

    return cells, cell_properties

def egg_half_IC(N : int) -> jnp.ndarray:

    cells, _ = egg_IC(N)
    cell_properties = jnp.where(cells[:,0,:][:,2] < 0, 0., 1.)

    return cells, cell_properties

def egg_genius(N:int) -> jnp.ndarray:
    cells, _ = egg_IC(N)

    cell_properties = jnp.where(cells[:,0,:][:,2] < 0, 0., 1.)

    cell_properties = jnp.where(cells[:,0,:][:,0] < -egg_shape[0]/2, 1., cell_properties)

    return cells, cell_properties

def egg_genius2(N:int) -> jnp.ndarray:
    cells, _ = egg_IC(N)

    cell_properties = jnp.where(cells[:,0,:][:,2] < egg_shape[2]/4, 0., 1.)

    cell_properties = jnp.where(cells[:,0,:][:,0] < -egg_shape[0]/2, 1., cell_properties)

    return cells, cell_properties

def better_egg(N:int) -> jnp.ndarray:
    cells, cell_properties = egg_IC(N)

    cells = make_better_egg(cells)
    return cells, cell_properties

def better_egg_genius(N:int) -> jnp.ndarray:
    cells, cell_properties = egg_IC(N)

    cells = make_better_egg(cells)
    cell_properties = jnp.where(cells[:,0,:][:,2] < egg_shape[2]/2, 0., 1.)

    cell_properties = jnp.where(cells[:,0,:][:,0] < -egg_shape[0]/2, 1., cell_properties)

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


def get_ic(type:str) -> jnp.ndarray:
    if type == "egg":
        return egg_IC
    elif type == "plane":
        return plane_IC
    elif type == "egg_half":
        return egg_half_IC
    elif type == "egg_genius":
        return egg_genius
    elif type == "egg_genius2":
        return egg_genius2
    elif type == "better_egg":
        return better_egg
    elif type == "continue":
        return continue_IC
    else:
        raise ValueError("type must be egg, plane or egg_half")
