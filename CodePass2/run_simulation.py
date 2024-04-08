from reworked import run_simulation
from torch import float as torch_float

l0 = 0.
l1, l2, l3 = 0.6, 0.1, 0.1
alpha = 0.2




# REMEMBER ADD MORE L3 FOR INNER

# interaction_data = [[l0, l1*0.8, l2, 0., 0.],        # green
#                     [l0, l1-l3-0.0, l2, l3, 0.],        # red
#                     [l0, l1-0.05*l3, l2, 0.05*l3, alpha*0.7],     # blue
#                     [l0, l1, l2, 0., -alpha*0.75],        # yellow
#                     [0.1, l1, l2, 0., alpha*0.42],     # baby blue
#                     [l0, l1+0.1, l2, 0., alpha*0.45] # pink
#                     ]


l1, l2, l3 = 1.0, 0.0, 0.0

interaction_data = [[l0, l1, l2, 0., 0.],        # green
                    [l0, l1, l2, 0., 0.],
                    [l0, l1, l2, 0., 0.],
                    [l0, l1, l2, 0., 0.],
                    [l0, l1, l2, 0., 0.],
                    [l0, l1, l2, 0., 0.]]


# interaction_data = [[0., 0.7, 0., 0., 0.],        # green
#                     [l0, l1, l2, l3, 0.],        # red
#                     [0.05, l1, l2, 0., alpha],     # blue
#                     [0., 0., 0., 0., 0.],        # yellow
#                     [0.05, 2*l1, l2, 0., 0.5*alpha],     # baby blue  
#                     ]



if __name__ == '__main__':
    sim_dict = {
        # 'data'              : (make_random_sphere, [size, conc, 35]), #if tuple 2 long then data generation, if 4 long then data
        # 'continue'          : "random_baby_2",
        'continue'          : "random_sphere",
        "name"              : "random_baby_3",
        'dt'                : 0.1,
        'eta'               : 0.002,
        'init_k'            : 100,
        'alpha'             : 0,
        'yield_steps'       : 200,
        'yield_every'       : 200,
        'interaction_data'  : interaction_data,
        'gamma'             : 5.0,
        'seethru'           : 1,
        'device'            : 'cuda',
        'dtype'             : torch_float,
        'notes'             : f"less green more l3",
    }

    run_simulation(sim_dict)