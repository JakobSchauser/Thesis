import sys
import importlib
import threading

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python parameter_scan.py <name> <parameter> <[steps]>")
        sys.exit(1)
    
    
    # assert sys.argv[2] in G, f"Parameter '{sys.argv[2]}' not found"

    steps = sys.argv[4].split(",")

    ts = []

    for step in steps:
        import simulate
        simulate = importlib.reload(simulate)
        G = simulate.G

        G["N_steps"] = int(sys.argv[3])
        G["name"] = sys.argv[1]
        s = float(step)
        G[sys.argv[2]] = s
        G["name"] = G["name"] + f"_{sys.argv[1]}_{sys.argv[2]}_{s}"
    
        # simulate.main(G["N_cells"], G["N_steps"], "new")

        t = threading.Thread(target=simulate.main, args=(G["N_cells"], G["N_steps"], "new"))
        t.start()
        ts.append(t)

    for t in ts:
        t.join()

    print("Done with all steps")


