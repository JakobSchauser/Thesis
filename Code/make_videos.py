# importing required libraries 
from matplotlib import pyplot as plt 
import numpy as np 
import matplotlib.animation as animation 
from IPython import display 
import h5py
import os


def make_videos(filename:str, video_name:str):
    # import the data
    print('Loading data')
    assert os.path.isfile('runs/' + filename + '.hdf5'), "Please provide a valid run filename"

    size = 85

    limits = 30

    with h5py.File('runs/' + filename + '.hdf5', 'r') as f:
        dat = f['cells'][:]
    positions = dat[:, :, 0]

    # make a 3d plot of the data
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


    # make an animation of the data
    def animate(i):
        ax.clear()
        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.set_zlim(-limits, limits)

        x, y ,z = positions[i, :, 0], positions[i, :, 1], positions[i, :, 2]

        # plot white dots with black outlines
        ax.scatter(x, y, z, s=size,       c='w', edgecolors='k')

        ax.scatter(x, z,  y - 40, s=size, c='w', edgecolors='k')

        ax.scatter(x, -z, y + 40, s=size, c='w', edgecolors='k')

        # remove the axes
        ax.set_axis_off()

        # place the camera
        ax.view_init(0, 270)

    print('Making Stas animation')
    ani = animation.FuncAnimation(fig, animate, frames=positions.shape[0], interval=12)
    print('Saving Stas animation')
    ani.save(video_name+'.mp4')
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

        # plot white dots with black outlines
        ax.scatter(x[y > 0], y[y > 0], z[y > 0]-20, s=size,       c='w', edgecolors='k')

        ax.scatter(y[x > 0], x[x > 0], z[x > 0]+20, s=size,       c='w', edgecolors='k')


        # remove the axes
        ax.set_axis_off()

        # place the camera
        ax.view_init(0, 270)

    print('Making cross section animation')
    ani = animation.FuncAnimation(fig, animate, frames=positions.shape[0], interval=12)
    print('Saving cross section animation')
    ani.save(video_name + '_cross_sections.mp4')
    print('Done!\n')




import sys
if __name__ == '__main__':
    if len(sys.argv) == 3:
        make_videos(sys.argv[1], sys.argv[2])
    if len(sys.argv) == 2:
        make_videos(sys.argv[1], sys.argv[1])
    else:
        print('Please provide a filename and/or a video name')