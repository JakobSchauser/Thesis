import numpy as np
import vispy.scene
from vispy.scene import visuals
import scipy
import sys
from vispy import app, scene
from vispy.visuals.transforms import STTransform
import os

iterator = 0

def interactive_plot(loaded_cells, loaded_properties = None, alpha=10):
    print('Plotting')

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    
    # Load the data
    x = loaded_cells[:,0,:]
    ps = loaded_cells[:,1,:]

        
    print(x.shape)
    print(x)

    # Create scatter object and fill in the data
    scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter1.set_data(x, edge_width=0, face_color=(1, 1, 1, .5), size=2.5)

    # color so we can see the direction of the particles
    scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter2.set_data(x + ps/10, edge_width=0, face_color=(1, 0, 0, .5), size=2.5)
    
    
    # Add the scatter objects to the view
    view.add(scatter1)
    view.add(scatter2)


    sphere1 = visuals.Sphere(radius=1, method='latitude', parent=view.scene,
                               color=(0,0,1,0.5),   )
    
    sphere1.transform = STTransform(scale = (200, 200/3, 200/3))
    # We want to fly around

    
    view.camera = 'fly'


    #add coordinate system
    axis = visuals.XYZAxis(parent=view.scene)


    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()

def interactive_animate(loaded_cells, loaded_properties = None, alpha=10,
                     frame_every = 1, interval=1/30 ):
    print('Animating')

    # Load the data
    xs = loaded_cells[:,:,0,:]
    ps = loaded_cells[:,:,1,:]

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()                               

    size = 10

    # Create scatter object and fill in the data
    colors = [(1,0,0) for i in range(len(xs[0]))]
    if loaded_properties is not None:
        colors = [(0,1,0) if loaded_properties[i] == 0 else (1,0,0) for i in range(len(loaded_properties))]

    scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter1.set_data(xs[0], edge_width=0, face_color=(1, 1, 1, .5), size=size)

    # color so we can see the direction of the particles
    scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter2.set_data(xs[0] + ps[0]/5, edge_width=0, face_color=colors, size=size)

    view.add(scatter1)
    view.add(scatter2)



    def update(ev):
        global iterator
        iterator += 1
        scatter1.set_data(xs[int(iterator%len(xs))], edge_width=0, face_color=(1, 1, 1, .5), size=size)
        scatter2.set_data(xs[int(iterator%len(xs))] + ps[int(iterator%len(xs))]/10, edge_width=0, face_color=colors, size=size)

    timer = app.Timer(interval=interval)
    timer.connect(update)
    timer.start()

    # We want to fly around
    view.camera = 'fly'

    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()

# def export_gif(folder, timesteps, output_name, alpha=10,
#                view_particles=None):

#     # Getting the data
#     x_lst = []
#     for timestep in timesteps:
#         P = scipy.io.loadmat(folder + f'/t{timestep}.mat')                    
#         x = P['x']
#         x_lst.append(x)

#     mask = np.load(folder + '/p_mask.npy') 
#     x_lst = np.array(x_lst)

#     # Make the canvas
#     canvas = scene.SceneCanvas(keys='interactive', bgcolor='black',
#                             size=(1200, 800), show=True)

#     view = canvas.central_widget.add_view()
#     view.camera = 'arcball'
#     view.camera.distance = 70

#     # Create scatter object and fill in the data
#     scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
#     scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
#     scatter1.set_data(x_lst[0][mask==0] , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
#     scatter2.set_data(x_lst[0][mask==1] , edge_width=0, face_color=(1, 1, 1, .5), size=2.5)

#     # Add the scatter object to the view
#     if not view_particles:
#         view.add(scatter1)
#         view.add(scatter2)
#     else:
#         assert view_particles == "polar" or view_particles == "non_polar", "view_particles only takes arguments polar or non_polar"
#         if view_particles == 'polar':
#             view.add(scatter2)
#         if view_particles == 'non_polar':
#             view.add(scatter1)

#     output_filename = f'{output_name}.gif'

#     writer = imageio.get_writer(output_filename)
#     for i in range(len(timesteps)):
#         im = canvas.render()
#         writer.append_data(im)
#         x = x_lst[int(i) % np.max(timesteps)]
#         scatter1.set_data(x[mask==0] , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
#         scatter2.set_data(x[mask==1] , edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
#         view.camera.transform.rotate(1, axis=[0,0,1])
#     writer.close()


if __name__ == '__main__':
    # get command line argument

    if len(sys.argv) > 1:
        loaded_cells = np.load(sys.argv[1])
        loaded_properties = None
        # check if properties file exists
        if os.path.isfile(sys.argv[1][:-4] + '_properties.npy'):
            loaded_properties = np.load(sys.argv[1][:-4] + '_properties.npy')

        if len(loaded_cells.shape) == 4:
            if len(sys.argv) > 2:
                if sys.argv[2] == '0':
                    interactive_plot(loaded_cells[0], loaded_properties)
                else:
                    interactive_animate(loaded_cells, loaded_properties)
            else:    
                interactive_animate(loaded_cells, loaded_properties)
        else:
            interactive_plot(loaded_cells, loaded_properties)
    else:
        print('No file specified')