import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

tracked_and_rescaled = pd.read_csv('tracked_and_rescaled.csv')

def get_scores(x0, x1, time_index, N_closest = 15):

    sim_d = (x1 - x0)
    sim_poss = x0

    scores = np.empty(sim_poss.shape[0])
    for i in range(sim_poss.shape[0]):
        print(f"{int(i/sim_poss.shape[0]*100)}%", end = '\r')

        data = tracked_and_rescaled[tracked_and_rescaled.frame == time_index]

        # calculate the distance between sp and all points in tracked_and_rescaled
        d = cdist([sim_poss[i]], data[['x', 'y', 'z']].values, 'euclidean')
        # find the closest points
        closest = np.argsort(d)[0][:N_closest]

        # get average dx, dy, dz in data
        dxs = data['dx'].iloc[closest].dropna()
        dys = data['dy'].iloc[closest].dropna()
        dzs = data['dz'].iloc[closest].dropna()

        if all(np.isnan(dxs)) or all(np.isnan(dys)) or all(np.isnan(dzs)):
            continue

        dx = dxs.mean()
        dy = dys.mean()
        dz = dzs.mean()
        
        # print("Here")
        # # compute the score
        # print(sim_d[i])
        # print(dx, dy, dz,)
        # print(dxs.std(), dys.std(), dzs.std())

        if np.isnan(dx) or np.isnan(dy) or np.isnan(dz):
            continue

        data_d_vector = np.array([dx, dy, dz])
        sim_d_vector = sim_d[i]

        # normalize
        data_d_vector = data_d_vector / np.linalg.norm(data_d_vector)
        sim_d_vector = sim_d_vector / np.linalg.norm(sim_d_vector)

        # compute the score
        score = np.dot(data_d_vector, sim_d_vector)

        # score = np.mean([np.abs(dx), np.abs(dy), np.abs(dz)])
        # score = np.mean([np.abs(sim_d[i][0] - dx), np.abs(sim_d[i][1] - dy), np.abs(sim_d[i][2] - dz)])
        scores[i] = score

    return scores

