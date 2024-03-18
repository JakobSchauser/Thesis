from reworked import run_simulation
from torch import float as torch_float

if __name__ == '__main__':


    sim_dict = {
        # 'data'              : (make_random_sphere, [size, conc, 35]), #if tuple 2 long then data generation, if 4 long then data
        'continue'          : "first_new_egg",
        "name"              : "first_run",
        'dt'                : 0.2,
        'eta'               : 0.2,
        'init_k'            : 100,
        'yield_steps'       : 2,
        'yield_every'       : 10_000,
        'lambdas'           : [0.3,0.3, 0.45, 0.0, 0.0],
        'gamma'             : 5.0,
        'seethru'           : 1,
        'alpha'             : 0,
        'device'            : 'cpu',
        'dtype'             : torch_float,
        'notes'             : f"Trying the new method",
    }

    run_simulation(sim_dict)