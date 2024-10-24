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



def make_videos(filename:str, interval :int):
    # import the data
    print('Loading data')
    # assert os.path.isfile('D:/' + filename + '.hdf5'), "Please provide a valid run filename"

    # if the video already exists, ask if the user wants to overwrite it
    if os.path.isfile(filename + '.mp4'):
        print('The video already exists, do you want to overwrite it? (y/n)')
        if input() != 'y':
            video_name = input('Please provide a new name for the video: ')
        else:
            video_name = filename
    else:
        video_name = filename

    size = 85

    limits = 45

    dists = 55
    has_energies = False

    # with h5py.File('D:/' + filename + '.hdf5', 'r') as f:
    #     dat = f['x'][::5]
    #     properties = f['properties'][:][0]
    #     pcp = f['q'][:][::5]
    positions = np.load("ffp.npy")[::5]
    properties = np.load("ffproperties.npy")
    # positions = dat[:,]
    # positions = dat


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

    colors = np.ones((positions[0].shape[0], 4))

    colors[properties == 0] = [1,1,1,1]
    colors[properties == 1] = [.7,.7,.7,1]
    colors[properties == 2] = [.8,.8,1.,1]
    colors[properties == 3] = [.8, 1,.8,1]
    colors[properties == 4] = [.8, 1,.8,1]
    colors[properties == 5] = [1,.8,.8,1]

    # make an animation of the data
    def animate(i):
        ax.clear()
        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.set_zlim(-limits, limits)

        x, y ,z = positions[i, :, 0], positions[i, :, 1], positions[i, :, 2]
      

        # plot white dots with black outlines
        ax.scatter(x, y, z, s=size,       c = colors, edgecolors='k')

        ax.scatter(x, z,  y - dists, s=size, c=colors, edgecolors='k')

        ax.scatter(x, -z, y + dists, s=size, c=colors, edgecolors='k')


        # closey = y<0
        # ax.quiver(x[closey], y[closey], z[closey], pcp[i, :, 0][closey], pcp[i, :, 1][closey], pcp[i, :, 2][closey], length=2, normalize=True, color='r')

        # closez = z < 0
        # ax.quiver(x[closez], z[closez], y[closez] - dists, pcp[i, :, 0][closez], pcp[i, :, 2][closez], pcp[i, :, 1][closez], length=2, normalize=True, color='r')

        # closez = z > 0
        # ax.quiver(x[closez], -z[closez], y[closez] + dists, pcp[i, :, 0][closez], pcp[i, :, 2][closez], pcp[i, :, 1][closez], length=2, normalize=True, color='r')
 
        ax.set_axis_off()

        print(i/positions.shape[0], end='\r')

        # place the camera
        ax.view_init(0, 270)

    print('Making Stas animation')
    ani = animation.FuncAnimation(fig, animate, frames=positions.shape[0], interval=interval)
    ani.save(video_name+'.mp4')
    print('Done!\n')


    # make a 3d plot of the data
    fig, ax = plt.subplots(figsize = (10,10))

    
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
        print(i/positions.shape[0], end='\r')

        # add image to the plot


 
    # print('Making voronoi animation')
    # ani = animation.FuncAnimation(fig, animate, frames=positions.shape[0], interval=12)
    # ani.save(video_name+'_voronoi.mp4')
    # print('Done!\n')


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

        ax.scatter(y[x > 10], x[x > 10], z[x > 10]+dists/2, s=size,       c='w', edgecolors='k')

        print(i/positions.shape[0], end='\r')

        # remove the axes
        ax.set_axis_off()

        # place the camera
        ax.view_init(0, 270)

    print('Making cross section animation')
    ani = animation.FuncAnimation(fig, animate, frames=positions.shape[0], interval=interval)
    ani.save(video_name + '_cross_sections.mp4')
    print('Done!\n')




import sys
if __name__ == '__main__':
    if len(sys.argv) == 3:
        make_videos(sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) == 2:
        make_videos(sys.argv[1], 12)
    else:
        print('Please provide a filename and/or a video name')