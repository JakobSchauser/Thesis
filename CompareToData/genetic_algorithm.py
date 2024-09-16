from reworked import genetic_step
from torch import float as torch_float
import numpy as np
from comparison_functions import get_scores
import matplotlib.pyplot as plt
import h5py

l0 = 0.
l1, l2, l3 = 1.0, 0.1, 0.2
alpha = 0.2

interaction_data = [[l0, l1, l2, 0., 0.],        # green
                    [l0, l1, l2, l3, 0.],
                    [l0, l1, l2, 0., alpha],
                    [l0, l1, l2, 0., 0.],
                    [l0, l1, l2, 0., 0.],
                    [l0, l1, l2, 0., 0.]]


continue_from = "stas_no_constrict_under_pole"

sim_dict = {
    # 'data'              : (make_random_sphere, [size, conc, 35]), #if tuple 2 long then data generation, if 4 long then data
    # 'continue'          : "random_baby_2",
    'continue'          : continue_from,
    "name"              : "genetic_test_1",
    'dt'                : 0.1,
    'eta'               : 0.002,
    'init_k'            : 100,
    'alpha'             : 0,
    'yield_steps'       : 100,
    'yield_every'       : 50,
    'interaction_data'  : interaction_data,
    'gamma'             : 5.0,
    'seethru'           : 1,
    'device'            : 'cuda',
    'dtype'             : torch_float,
    'notes'             : f"trying stripes",
}


with h5py.File("runs/"+continue_from+".hdf5", "r") as f:
    if len(f['x'][:].shape) > 2:
        x = f['x'][-1]
        p = f['p'][-1]
        q = f['q'][-1]
    else:
        x = f['x'][:]
        p = f['p'][:]
        q = f['q'][:]
        
    if len(f['properties']) == 1:
        properties = np.array(f['properties'][0])
    else:
        properties = np.array(f['properties'])

startx = x[-1]

def from_pos_to_perc(poss):
    xx = poss[-1][:,0]
    yy = poss[-1][:,1]
    zz = poss[-1][:,2]

    xx = (xx - xx.min())/(xx.max() - xx.min())*100
    yy = (yy - yy.min())/(yy.max() - yy.min())*100
    zz = (zz - zz.min())/(zz.max() - zz.min())*100

    return np.array([xx, yy, zz]).T

N_cell_types = 2


def get_genetics(cuts, percs):
    genetics = np.zeros(5000)
    xx,yy,zz = percs.T

    for i in range(N_cell_types):
        cut = (xx > cuts[i,0,0]) * (xx < cuts[i,0,1]) * (yy > cuts[i,1,0]) * (yy < cuts[i,1,1]) * (zz > cuts[i,2,0]) * (zz < cuts[i,2,1])
        genetics[cut] = (i+1) 
    
    return genetics


if __name__ == "__main__":
    N_steps = 10
    population_size = 10
    cuts = [np.random.randint(0, 100, (N_cell_types, 3, 2)) for _ in range(population_size)]



    for i in range(N_steps):
        scores = np.zeros(population_size)
        for j in range(population_size):
            print("Making genetics")
            print("Current cuts are:")
            print(cuts[j])

            genetics = get_genetics(cuts[j], from_pos_to_perc([x]))

            print("Simulating")
            Xs = genetic_step(sim_dict, genetics)

            print("Getting scores")
            score  = np.mean(get_scores(startx, Xs, 1, N_closest = 25))

            scores[j] = score
            print("Final Score:")
            print(score)
        
        argsorted = np.argsort(scores)
        cuts = [cuts[index] for index in argsorted]
        parents = cuts[population_size//2:]
        print(f"Step {i}, top scorers score: {np.array(scores)[argsorted][population_size//2:]}")


        # save best to txt file
        with open("best_cuts.txt", "a") as f:
            f.write(f"Step {i}, top scorers score: {np.array(scores)[argsorted][population_size//2:]}\n")
            f.write(f"Step {i}, top scorers cuts: {np.array(cuts)[argsorted][population_size//2:]}\n")


        children = []
        for parent in parents:
            child = parent.copy()
            mutation = np.random.randint(-10, 10, (N_cell_types, 3, 2))
            child += mutation
            children.append(child)
        
        cuts = parents + children







