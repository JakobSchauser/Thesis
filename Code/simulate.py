import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

from jax import random

from jax.config import config

from jax_tqdm import loop_tqdm

import functools

config.update("jax_debug_nans", False)


from functools import cache

import os

from InitialConditions import InitialConditions

import h5py

import sys 

# from jax_enums import Enumerable
import enum

class BC(enum.IntEnum):
    NONE = 0
    SPHERE = 1
    EGG = 2
    BETTER_EGG = 3
    EVEN_BETTER_EGG = 4

class S_type(enum.IntEnum):
    ONLY_AB = 0
    STANDARD = 1
    ANGLE = 2
    WEAK_AB = 3
    WEAK_STANDARD = 4
    INVERSE_ANGLE = 5
    ANGLE_ISOTROPIC = 6
    NON_INTERACTING = 7
    WEAKER_AB = 8

interaction_matrix = jnp.array([
                                [S_type.ONLY_AB, S_type.WEAKER_AB, S_type.WEAKER_AB, S_type.ONLY_AB, S_type.WEAKER_AB, S_type.ONLY_AB, S_type.ANGLE_ISOTROPIC, S_type.NON_INTERACTING, S_type.WEAKER_AB], 
                                [S_type.WEAKER_AB, S_type.STANDARD, S_type.STANDARD, S_type.WEAKER_AB, S_type.STANDARD, S_type.STANDARD, S_type.ANGLE_ISOTROPIC, S_type.NON_INTERACTING,  S_type.WEAKER_AB],
                                [S_type.ONLY_AB, S_type.STANDARD, S_type.ANGLE, S_type.WEAKER_AB, S_type.WEAKER_AB, S_type.WEAK_AB, S_type.ANGLE_ISOTROPIC, S_type.NON_INTERACTING,  S_type.WEAKER_AB],
                                [S_type.WEAK_AB, S_type.WEAKER_AB, S_type.WEAKER_AB, S_type.WEAK_AB, S_type.WEAKER_AB ,S_type.WEAK_AB, S_type.ANGLE_ISOTROPIC, S_type.NON_INTERACTING,  S_type.WEAKER_AB],
                                [S_type.WEAKER_AB, S_type.WEAK_STANDARD, S_type.WEAK_STANDARD, S_type.WEAKER_AB, S_type.WEAK_STANDARD,S_type.WEAK_STANDARD, S_type.ANGLE_ISOTROPIC, S_type.NON_INTERACTING,  S_type.WEAKER_AB],
                                [S_type.ONLY_AB, S_type.STANDARD, S_type.INVERSE_ANGLE, S_type.WEAK_AB, S_type.WEAK_STANDARD, S_type.INVERSE_ANGLE, S_type.ANGLE_ISOTROPIC, S_type.NON_INTERACTING,  S_type.WEAKER_AB],#last one should be inverse
                                [S_type.ANGLE_ISOTROPIC, S_type.ANGLE_ISOTROPIC, S_type.ANGLE_ISOTROPIC, S_type.ANGLE_ISOTROPIC, S_type.ANGLE_ISOTROPIC, S_type.ANGLE_ISOTROPIC, S_type.ANGLE_ISOTROPIC, S_type.NON_INTERACTING,  S_type.WEAKER_AB],
                                [S_type.NON_INTERACTING, S_type.NON_INTERACTING, S_type.NON_INTERACTING, S_type.NON_INTERACTING, S_type.NON_INTERACTING, S_type.NON_INTERACTING, S_type.NON_INTERACTING, S_type.NON_INTERACTING,  S_type.NON_INTERACTING],
                                [S_type.WEAKER_AB,  S_type.WEAKER_AB,  S_type.WEAKER_AB,  S_type.WEAKER_AB,  S_type.WEAKER_AB,  S_type.WEAKER_AB,  S_type.WEAKER_AB,  S_type.NON_INTERACTING,  S_type.WEAKER_AB]])

@jit
def unpack_cellrow(cellrow : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    pos = cellrow[0]
    p = cellrow[1]
    q = cellrow[2]
    return pos, p, q

@jit
def quadruple(a1, a2, b1, b2) -> float: 
    return jnp.dot(jnp.cross(a1, b1), jnp.cross(a2, b2))

@jit
def V(r : float, S : float, A : float) -> float:
    return jnp.exp(-r/A) - S * jnp.exp(- r / G["beta"])

@jit
def S_standard(r, p1, q1, p2, q2) -> float:
    S1 = quadruple(p1, p2, r, r)
    S2 = quadruple(p1, p2, q1, q2)
    S3 = quadruple(q1, q2, r, r)
    return G["lambda1"]*S1 + G["lambda2"]*S2 + G["lambda3"]*S3

@jit
def S_standard_weak(r, p1, q1, p2, q2) -> float:
    return S_standard(r, p1, q1, p2, q2)*0.5
@jit
def S_only_AB(r, p1, q1, p2, q2) -> float:
    S1 = quadruple(p1, p2, r, r)
    S2 = quadruple(p1, p2, q1, q2)

    return S1*0.6 + S2*0.4

@jit
def S_only_AB_weak(r, p1, q1, p2, q2) -> float:
    S1 = quadruple(p1, p2, r, r)
    S2 = quadruple(p1, p2, q1, q2)

    return S1*0.4 + S2*0.4

@jit
def S_only_AB_weaker(r, p1, q1, p2, q2) -> float:
    S1 = quadruple(p1, p2, r, r)
    S2 = quadruple(p1, p2, q1, q2)

    return S1*0.2 + S2*0.4
    
@jit
def S_angle(r, p1, q1, p2, q2) -> float:
    avg_q = (q1 + q2)*0.5        #TODO: average q on unit sphere?

    r_unit = r / jnp.linalg.norm(r)

    phat1 = p1 + G["alpha"]*avg_q*jnp.sum(avg_q*r_unit)
    phat1 = phat1 / jnp.linalg.norm(phat1)

    phat2 = p2 - G["alpha"]*avg_q*jnp.sum(avg_q*r_unit)
    phat2 = phat2 / jnp.linalg.norm(phat2)


    S1 = quadruple(phat1, phat2, r, r)
    S2 = quadruple(p1, p2, q1, q2)

    return (0.6*S1 + 0.4*S2)

@jit
def S_angle_isotropic(r, p1, q1, p2, q2) -> float:

    r_unit = r / jnp.linalg.norm(r)

    phat1 = p1 + G["alpha"]*r_unit
    phat1 = phat1 / jnp.linalg.norm(phat1)

    phat2 = p2 - G["alpha"]*r_unit
    phat2 = phat2 / jnp.linalg.norm(phat2)


    S1 = quadruple(phat1, phat2, r, r)
    S2 = quadruple(p1, p2, q1, q2)

    return (0.6*S1 + 0.4*S2)*1.


@jit
def S_inverse_angle(r, p1, q1, p2, q2) -> float:
    avg_q = (q1 + q2)*0.5        #TODO: average q on unit sphere?

    r_unit = r / jnp.linalg.norm(r)

    phat1 = p1 + G["alpha"]*avg_q*jnp.sum(avg_q*r_unit)*0.5
    phat1 = phat1 / jnp.linalg.norm(phat1)

    phat2 = p2 - G["alpha"]*avg_q*jnp.sum(avg_q*r_unit)*0.5
    phat2 = phat2 / jnp.linalg.norm(phat2)


    S1 = quadruple(phat1, phat2, r, r)
    S2 = quadruple(p1, p2, q1, q2)
    S3 = quadruple(q1, q2, r, r)

    return (1 - 0.5 - 0.05)*S1 + 0.5*S2 + 0.05*S3

@jit
def S_non_interacting(r, p1, q1, p2, q2) -> float:
    return 1.



def from_angles_to_vector(phi, theta) -> jnp.ndarray:
    x = jnp.cos(phi)*jnp.sin(theta)
    y = jnp.sin(phi)*jnp.cos(theta)
    z = jnp.sin(phi)
    return jnp.array([x, y, z])


@cache
def get_boundary_fn():
    boundary = G["boundary"]
    if boundary == BC.NONE:
        def none(pos):
            mag = jnp.abs(jnp.dot(pos, pos)) # squared length
            return jnp.where(mag < -1.0, 0., 0.)
        return none
    elif boundary == BC.SPHERE:
        def sphere(pos):
            corrected_pos_vec = pos / G["IC"].scaled_sphere_shape
            mag = jnp.dot(corrected_pos_vec, corrected_pos_vec) # squared length
            v_add = jnp.where(mag > 1.0, (jnp.exp(mag*mag - 1) - 1), 0.)
            return v_add 
        return sphere
    elif boundary == BC.EGG:
        def egg(pos):
            corrected_pos_vec = pos / G["IC"].scaled_egg_shape
            mag = jnp.dot(corrected_pos_vec, corrected_pos_vec) # squared length
            v_add = jnp.where(mag > 1.0, (jnp.exp(mag*mag - 1) - 1), 0.)
            return v_add 
        return egg
    elif boundary == BC.BETTER_EGG:
        def better_egg(pos):
            corrected_pos_vec = G["IC"].inv_make_better_egg_pos(pos)
            corrected_pos_vec = corrected_pos_vec / G["IC"].scaled_egg_shape
            mag = jnp.dot(corrected_pos_vec, corrected_pos_vec) # squared length
            v_add = jnp.where(mag > 1.0, jnp.minimum((jnp.exp(mag*mag - 1) - 1), 5.), 0.)
            return v_add 
        return better_egg
    elif boundary == BC.EVEN_BETTER_EGG:
        def even_better_egg(pos):
            corrected_pos_vec = G["IC"].inv_make_even_better_egg_pos(pos)
            corrected_pos_vec = corrected_pos_vec / G["IC"].scaled_egg_shape
            mag = jnp.dot(corrected_pos_vec, corrected_pos_vec) # squared length
            v_add = jnp.where(mag > 1.0, jnp.minimum((jnp.exp(mag*mag - 1) - 1), 5.), 0.)
            return v_add 
        return even_better_egg
    else:
        raise Exception("boundary must be none, sphere, egg or better_egg")

# def get_S_fn(prop : int):
#     property = G["cell_properties"][prop]
#     if property == S_interact.STANDARD:
#         return S_standard
#     elif property == S_interact.ONLY_AB:
#         return S_only_AB
#     else:
#         raise Exception("cell_properties must be standard or only_AB")




def get_interaction(prop1 : int, prop2 : int, *args):
    interact = interaction_matrix.at[prop1, prop2].get()
    return jax.lax.switch(interact, [S_only_AB, S_standard, S_angle, S_only_AB_weak, S_standard_weak, S_inverse_angle, S_angle_isotropic, S_non_interacting, S_only_AB_weaker], *args)


def get_shape_change(q1, q2, dir, type1 : int):
    lf = lambda p1, p2, d1, d2: 0.0

    l2 = lambda q1, q2, dir1, dir2: jnp.abs(jnp.dot(q1, dir1))

    A = jax.lax.switch(type1, [lf, lf, lf, lf, lf, l2], q1, q2, dir, dir)

    return 0.8 + A*0.3

@jit
def U(cellrow1 : jnp.ndarray, cellrow2 : jnp.ndarray, cell1_property : float, cell2_property : float) -> float:
    pos1, p1, q1 = unpack_cellrow(cellrow1)
    pos2, p2, q2 = unpack_cellrow(cellrow2)

    dir = pos1 - pos2
    _dir = jnp.where(jnp.allclose(dir, jnp.zeros(3)), jnp.array([1, 0, 0]), dir)

    norm_dir = _dir/jnp.linalg.norm(_dir)

    # s = S(dir, p1, q1, p2, q2)
    # s = jax.lax.cond(cell2_property == 0.0, G["cell_properties"]["standard"]["S"], G["cell_properties"]["non_polar"]["S"], norm_dir, p1, q1, p2, q2)
    # s_type = jax.lax.switch(cell2_property, G["cell_properties"])
    s1 = G["cell_properties"].at[cell1_property.astype(int)].get()
    s2 = G["cell_properties"].at[cell2_property.astype(int)].get()
    # s1 = jax.lax.switch(, )
    # s2 = jax.lax.switch(cell2_property.astype(int), G["cell_properties"])

    s = get_interaction(s1, s2, norm_dir, p1, q1, p2, q2)

    r = jnp.linalg.norm(_dir)

    A = get_shape_change(q1, q2, _dir, cell1_property.astype(int))

    v = V(r, s, A)

    # add the boundary
    boundary_fn = get_boundary_fn()
    v_add = boundary_fn(pos1)

    v_added = v + v_add

    # correct for the case where the particles are on top of each other
    v_corrected = jnp.where(jnp.allclose(dir, jnp.zeros(3)), 0.0, v_added)

    return v_corrected


@cache
def find_array_that_has_own_indexes(size):
    indx = jnp.repeat(jnp.arange(size[0]), repeats = size[1]).reshape((size[0], size[1]))
    return indx

@functools.partial(jit, static_argnames=["k", "recall_target"])
def l2_ann(qy, db,  k=10, recall_target=0.95):
    half_db_norms = jnp.linalg.norm(db, axis=1)**2 / 2
    dists = half_db_norms - jax.lax.dot(qy, db.transpose())
    return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)

@functools.partial(jit, static_argnames=["k"])
def find_neighbors(cells : jnp.ndarray, k : int = 10):
    positions = cells[:,0,:]
    dists, neighbors = l2_ann(positions, positions, k=k+1)

    neighbors = neighbors[:,1:] # remove the first neighbor, which is the particle itself

    neighbor_positions = positions[neighbors]  # (N,k,3)

    dx = neighbor_positions - positions[:,None,:]

    d = jnp.linalg.norm(dx, axis=2)

    # # only find those that can interact through voronoi
    n_dis = jnp.sum((dx[:, :, None, :] / 2 - dx[:, None, :, :]) ** 2, axis=3)    # Finding distances from particle k to the midpoint between particles i and j squared for all potential neighbors
    n_dis += 1000 * jnp.eye(n_dis.shape[1])[None, :, :]                  # We add 1000 to the sites that correspond to subtracting the vector ij/2 with ij, so these don't fuck anything up.

    z_mask = jnp.sum(n_dis < (d[:, :, None] ** 2 / 4), axis=2) <= 0

    indx = find_array_that_has_own_indexes((neighbors.shape[0], k))
    # indx = jnp.indices((neighbors.shape[0], k))
    actual_neighbors = jnp.where(z_mask, neighbors, indx)

    return actual_neighbors


# compute the energy of the system
@functools.partial(jit, static_argnames=['cell_properties'])
def U_sum(cells : jnp.ndarray, neighbors : jnp.ndarray, cell_properties : jnp.ndarray) -> float:

    # save the array
    arr = jnp.empty((cells.shape[0], neighbors.shape[1]), float)

    def loop_fn(i, arr):
        val = jax.vmap(lambda nb, prop: U(cells[i], nb, cell_properties[i], prop))(cells[neighbors[i,:]], cell_properties[neighbors[i,:]])
        arr = arr.at[i,:].set(val)
        return arr
    
    arr = jax.lax.fori_loop(0, cells.shape[0], loop_fn, arr)

    energies = np.sum(arr, axis = 1)

    final_sum = jnp.sum(energies)

    return final_sum, energies / jnp.count_nonzero(arr, axis = 1)

U_grad = grad(U_sum, argnums=(0), has_aux=True)


def take_step(step_indx : int, cells : jnp.ndarray, old_nbs : jnp.ndarray, cell_properties : jnp.ndarray,):

    neighbors = jax.lax.cond((step_indx < 30) | (step_indx % 50 == 0), find_neighbors, lambda *args: old_nbs, cells)

    grad_U, energies = U_grad(cells, neighbors, cell_properties)

    # update the positions using euler
    cells = cells - grad_U*G["dt"]

    # add random noise to the positions
    cells = cells.at[:,0,:].add(random.normal(random.PRNGKey(0), cells.shape[0:2])*G["eta"])

    # normalize p and q
    ns = cells[:,1:,:] / jnp.linalg.norm(cells[:,1:,:], axis=2)[:,:,None]
    cells = cells.at[:,1:,:].set(ns)
    

    return cells, neighbors, energies

take_step = jit(take_step, static_argnames=["cell_properties"])


def get_save_fn(name : str, type : str):
    def save_file(name, all_cells, cell_properties, all_energies, G):
        with h5py.File("runs/"+name+".hdf5", "w") as f:
            f.create_dataset("cells", data=all_cells)
            if G["save_energies"]:
                f.create_dataset("energies", data=all_energies)
            f.create_dataset("properties", data=cell_properties)
            f.attrs.update(G_to_properties(G))

    if type == "append":
        def save_fn(all_cells, cell_properties, all_energies, G):
            with h5py.File("runs/"+name+".hdf5", "r") as f:
                old_cells = f["cells"][:]
                if G["save_energies"]:
                    old_energies = f["energies"][:]
                old_prop = dict(f.attrs.items())


            appended_cells = np.concatenate((old_cells, all_cells), axis=0)
            if G["save_energies"]:
                appended_energies = np.concatenate((old_energies, all_energies), axis=0)

            prop = G_to_properties(G)
            old_prop["N_steps"] = old_prop["N_steps"] + G["N_steps"]
            old_prop["cell_properties"] = prop["cell_properties"]
            
            with h5py.File("runs/"+name+".hdf5", "a") as f:
                del f["cells"]
                f.create_dataset("cells", data=appended_cells)

                if G["save_energies"]:
                    del f["energies"]
                    f.create_dataset("energies", data=appended_energies)
                f.attrs.update(old_prop)

        return save_fn
    elif type == "branch":
        def save_fn(all_cells, cell_properties, all_energies, G):
            with h5py.File("runs/"+name+".hdf5", "r") as f:
                old_cells = f["cells"][:]
                if G["save_energies"]:
                    old_energies = f["energies"][:]
                old_prop = dict(f.attrs.items())

            appended_cells = np.concatenate((old_cells, all_cells), axis=0)

            if G["save_energies"]:
                appended_energies = np.concatenate((old_energies, all_energies), axis=0)

            prop = G_to_properties(G)
            prop["N_steps"] = old_prop["N_steps"] + G["N_steps"]

            save_file(G["new_name"], appended_cells, cell_properties, appended_energies, G)
        return save_fn
    else:
        def save_fn(all_cells, cell_properties, all_energies, G):
            if type == "overwrite":
                os.remove("runs/"+name+".hdf5")
            save_file(name, all_cells, cell_properties,all_energies, G)
        return save_fn

def main(N_cells : int, N_steps : int, save_type : str):
    save_every = G["save_every"]

    assert np.isclose(G["lambda1"] + G["lambda2"] + G["lambda3"], 1.0), "lambda1 + lambda2 + lambda3 must sum to 1.0 but is " + str(G["lambda3"] + G["lambda1"] + G["lambda2"])
        
    assert save_every > 0 and save_every < N_steps, "save_every must be positive but smaller than N_steps"

    if not (save_type == "continue" or save_type == "branch" or save_type == "append"):
        IC_cells, IC_cell_properties = G["IC"].initialize()
    else:
        with h5py.File("runs/"+G["name"]+".hdf5", "r") as f:
            IC_cells = np.array(f["cells"][-1])
            IC_cell_properties = np.array(f["properties"])
            old_G = dict(f.attrs.items())
        if save_type == "continue":
            G_from_properties(old_G)
            N_cells = G["N_cells"]
            G["N_steps"] = N_steps

    save_to_disk = get_save_fn(G["name"], save_type)

    cells = IC_cells
    cell_properties = IC_cell_properties


    print("starting simulation")

    all_cells = jnp.empty((int(N_steps/save_every), N_cells, 3, 3), float)
    all_energies = jnp.empty((int(N_steps/save_every), N_cells), float)
    energies = jnp.zeros((N_cells), float)
    old_nbs = jnp.empty((cells.shape[0], 10), int)#*-1   # why did I do this?

    def save_cells(i, cells, all_cells, save_every):
        all_cells = all_cells.at[jnp.floor(i/save_every).astype(int),:,:,:].set(cells)
        return all_cells
    
    def save_energies(i, energies, all_energies, save_every):
        all_energies = all_energies.at[jnp.floor(i/save_every).astype(int),:].set(energies)
        return all_energies
    
    @loop_tqdm(N_steps)
    def loop_fn(i, cp, save_every=save_every):
        cells, all_cells, old_nbs, cell_properties, all_energies, energies = cp
        all_cells = jax.lax.cond(i % save_every == 0, lambda : save_cells(i, cells, all_cells, save_every), lambda *args: all_cells)
        all_energies = jax.lax.cond(G["save_energies"] & (i % save_every == 0), lambda : save_energies(i, energies, all_energies, save_every), lambda *args: all_energies)

        cells, old_nbs, energies = take_step(i, cells, old_nbs, cell_properties)
        return cells, all_cells, old_nbs, cell_properties, all_energies, energies
    
    cells, all_cells, old_nbs, cell_properties, all_energies, energies = jax.lax.fori_loop(0, N_steps, loop_fn, (cells, all_cells, old_nbs, cell_properties, all_energies, energies))

    save_to_disk(all_cells, cell_properties, all_energies,  G)

    return cells


def G_to_properties(G):
    proplist = G["cell_properties"] if type(G["cell_properties"]) == list else G["cell_properties"].tolist()

    return {
        "alpha": G["alpha"],
        "beta": G["beta"],
        "dt": G["dt"],
        "eta": G["eta"],
        "lambda1": G["lambda1"],
        "lambda2": G["lambda2"],
        "lambda3": G["lambda3"],
        "boundary": G["boundary"] if type(G["boundary"]) == str else G["boundary"].name,
        "N_cells": G["N_cells"],
        "N_steps": G["N_steps"],
        "cell_properties": [prop if type(prop) == str else S_type(prop).name for prop in proplist],
        }

def G_from_properties(old_G):
    G["alpha"] = old_G["alpha"]
    G["beta"] = old_G["beta"]
    G["dt"] = old_G["dt"]
    G["eta"] = old_G["eta"]
    G["lambda1"] = old_G["lambda1"]
    G["lambda2"] = old_G["lambda2"]
    G["lambda3"] = old_G["lambda3"]
    G["boundary"] = BC[old_G["boundary"]]
    G["N_cells"] = old_G["N_cells"]
    G["N_steps"] = old_G["N_steps"]
    # G["cell_properties"] = [getattr(IC, prop) for prop in G["cell_properties"]]

# G = {
#     "N_steps": 500,
#     "alpha": 0.5,
#     "beta": 5.0,
#     "dt": 0.1,
#     "eta": 0.5e-4, # width of the gaussian noise
#     "lambda3": 0.12,
#     "lambda2": 0.5,
#     "lambda1": 1 - 0.5 - 0.12,
#     "boundary": BC.BETTER_EGG,   # none, sphere, egg, better_egg
#     "N_cells": 5000,
#     "cell_properties": jnp.array([S_type.WEAK_AB, S_type.WEAK_STANDARD, S_type.ANGLE, S_type.WEAK_AB, S_type.ANGLE_ISOTROPIC]),
#     "save_every": 20, # only used if save == 2
#     "save_energies": False,
#     "IC_scale" : 65.,
#     # "IC_scale" : 41.5,
#     "IC_type" : "continue:large_timeline_base12", # continue, plane, sphere, egg, better_egg
# }
    
G = {
"N_steps": 500,
"alpha": 0.5,
"beta": 5.0,
"dt": 0.1,
"eta": 0.5e-4, # width of the gaussian noise
"lambda3": 0.3,
"lambda2": 0.5,
"lambda1": 1 - 0.5 - 0.3,
"proliferate" : False,
"proliferation_rate" : 0.0, # per time step
# "max_cells" : 2000,
# "boundary": BC.BETTER_EGG,   # none, sphere, egg, better_egg
"boundary": BC.EVEN_BETTER_EGG,   # none, sphere, egg, better_egg, even_better_egg
"N_cells": 5000,
"cell_properties": jnp.array([S_type.ONLY_AB]),
"save_every": 1, # only used if save == 2
# "IC_scale" : 65.,
"save_energies": False,

# "IC_scale" : 41.5,
# "IC_scale" : 65,
"IC_scale" : 70,
"IC_type" : "continue:pink_top", # continue, plane, sphere, egg, better_egg, even_better_egg, ball
}

IC = InitialConditions(G)

G["IC"] = IC


import sys
if __name__ == '__main__':
    assert len(sys.argv) > 1, "must specify a name"

    if os.path.exists("runs/"+ sys.argv[1]+".hdf5"):
        assert len(sys.argv) > 2, "must specify a save_type, overwrite or append?"
        save_type = sys.argv[2]

        assert save_type in ["append", "overwrite", "branch", "continue"], "save_type must be append, continue, overwrite or branch"
        if save_type == "branch":
            assert len(sys.argv) > 3, "must specify a new name"
            G["new_name"] = sys.argv[3]
    else:
        assert len(sys.argv) < 3, "cannot specify a save_type if the file does not exist"
        save_type = "new"
    G["name"] = sys.argv[1]

    main(G["N_cells"], G["N_steps"], save_type)
