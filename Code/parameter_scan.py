import sys
import importlib
import threading

if __name__ == "__main__":

    arguments = sys.argv
    if len(arguments) < 4:
        print("Usage: python parameter_scan.py <name> <parameter> <steps> [<parameter> <steps>]")
        sys.exit(1)
    
    if len(arguments) >= 4:
        assert len(arguments) in [4,6,8]
    # assert arguments[2] in G, f"Parameter '{arguments[2]}' not found"

    N_params = (len(arguments)-2)//2

    all_parameters = []
    all_steps = []

    for i in range(N_params):
        all_parameters.append(arguments[3 + 2*i])
        all_steps.append(arguments[4 + 2*i].split(","))

    threads = []

    for parameter, steps in zip(all_parameters, all_steps):
        for step in steps:
            import simulate
            simulate = importlib.reload(simulate)
            G = simulate.G

            G["name"] = arguments[1]
            s = float(step)
            G[arguments[2]] = s
            G["name"] = G["name"] + f"_{arguments[1]}_{arguments[2]}_{s}"
            print(f"Starting {G['name']}")
            # simulate.main(G["N_cells"], G["N_steps"], "new")

            t = threading.Thread(target=simulate.main, args=(G["N_cells"], G["N_steps"], "new"))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    print("Done with all steps")


