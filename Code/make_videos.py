# importing required libraries 
from matplotlib import pyplot as plt 
import numpy as np 
import matplotlib.animation as animation 
from IPython import display 
import h5py
import os
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi
from utils import voronoi_plot_2d



def make_videos(filename:str, video_name:str):
    # import the data
    print('Loading data')
    assert os.path.isfile('runs/' + filename + '.hdf5'), "Please provide a valid run filename"

    size = 85

    limits = 45

    dists = 55
    has_energies = False

    with h5py.File('runs/' + filename + '.hdf5', 'r') as f:
        dat = f['cells'][:]
        properties = f['properties'][:]

        if "energies" in f.keys():
            energies = f['energies'][:]
            has_energies = True

    positions = dat[:, :, 0]


    # # make a 3d plot of the data
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d', )
    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    ax.set_zlim(-limits, limits)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.tight_layout()
    # plot the positions
    ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], s=size)

    # remove the axes
    ax.set_axis_off()

    # place the camera
    ax.view_init(0, 270)

    property_dists = np.where((properties == 1) | (properties == 3), 0.18, 0.52,)


    # make an animation of the data
    def animate(i):
        ax.clear()
        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.set_zlim(-limits, limits)

        x, y ,z = positions[i, :, 0], positions[i, :, 1], positions[i, :, 2]
        # x = x[z < 25]
        # y = y[z < 25]
        # z = z[z < 25]

        # all_dists = squareform(pdist(positions[i]))

        # # find the five closest points to each point

        # closest = np.argsort(all_dists, axis=1)[:, 1:6]

        # # find the mean distance to the closest points

        # mean_dist = np.mean(all_dists[np.arange(all_dists.shape[0])[:,None], closest], axis=1)
        # # color the points based on the mean distance to the three closest points



        # n = np.clip((property_dists-mean_dist)*2., -1., 1.)
        colors = np.ones((positions[i].shape[0], 4))

        # if has_energies:
        #     colors = np.zeros((positions[i].shape[0], 4))
            
        #     n = np.clip((property_dists + energies[i])*4., -1., 1.)

        #     colors[:, 0] = np.clip(1 + n, 0, 1)
        #     colors[:, 1] = np.minimum(1 - n, 1 + n)
        #     colors[:, 2] = np.clip(1 - n, 0., 1)
        #     colors[:, 3] = 1


        # plot white dots with black outlines
        ax.scatter(x, y, z, s=size,       c = colors, edgecolors='k')

        ax.scatter(x, z,  y - dists, s=size, c=colors, edgecolors='k')

        ax.scatter(x, -z, y + dists, s=size, c=colors, edgecolors='k')

 
        ax.set_axis_off()

        # place the camera
        ax.view_init(0, 270)

    print('Making Stas animation')
    ani = animation.FuncAnimation(fig, animate, frames=positions.shape[0], interval=8)
    ani.save(video_name+'.mp4')
    print('Done!\n')

    positions = dat[:, :, 0]


    # make a 3d plot of the data
    fig, ax = plt.subplots(figsize = (10,10))

    x, y ,z = positions[0, :, 0], positions[0, :, 1], positions[0, :, 2]

    
    # ax.set_xlim(np.min(x) - 2, np.max(x) + 2)
    # ax.set_ylim(np.min(X) - 2, np.max(X) + 2)
    ax.set_xlim(-65,65)
    ax.set_ylim(-65,65)
    ax.axis('off')


    # fig.tight_layout()
    # plot the positions

    # # add an image that fills the whole plot
    # im = plt.imread('egg_cutout.png')
    # newax = fig.add_axes([0, 0, 1, 1], anchor='C', zorder=999)
    # newax.axis('off')
    # newax.imshow(im, alpha=1, extent=[-65, 65, -65, 65])
    
    # make an animation of the data
    def animate(i):
        ax.clear()

        x, y ,z = positions[i, :, 0], positions[i, :, 1], positions[i, :, 2]
        # only get x and y 
        xy_coordinates = np.vstack((x[y < -3], z[y < -3])).T



        vor = Voronoi(xy_coordinates)
        voronoi_plot_2d(vor, show_vertices=False, line_width=2, line_alpha=0.6, point_size=10, ax = ax, point_alpha=0)
        
        ax.set_xlim(-65,65)
        ax.set_ylim(-65,65)
        ax.axis('off')

        # add image to the plot


 
    print('Making voronoi animation')
    ani = animation.FuncAnimation(fig, animate, frames=positions.shape[0], interval=8)
    ani.save(video_name+'_voronoi.mp4')
    print('Done!\n')


    # make a 3d plot of the data
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    ax.set_zlim(-limits, limits)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.tight_layout()

    # plot the positions
    ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], s=size)

    # remove the axes
    ax.set_axis_off()

    # place the camera
    ax.view_init(0, 270)

    # make an animation of the data
    def animate(i):
        ax.clear()
        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.set_zlim(-limits, limits)

        x, y ,z = positions[i, :, 0], positions[i, :, 1], positions[i, :, 2]
        x = x[z < 25]
        y = y[z < 25]
        z = z[z < 25]
        # plot white dots with black outlines
        ax.scatter(x[y > 0], y[y > 0], z[y > 0]-dists/2, s=size,       c='w', edgecolors='k')

        ax.scatter(y[x > 0], x[x > 0], z[x > 0]+dists/2, s=size,       c='w', edgecolors='k')


        # remove the axes
        ax.set_axis_off()

        # place the camera
        ax.view_init(0, 270)

    print('Making cross section animation')
    ani = animation.FuncAnimation(fig, animate, frames=positions.shape[0], interval=8)
    ani.save(video_name + '_cross_sections.mp4')
    print('Done!\n')




import sys
if __name__ == '__main__':
    if len(sys.argv) == 3:
        make_videos(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        make_videos(sys.argv[1], sys.argv[1])
    else:
        print('Please provide a filename and/or a video name')