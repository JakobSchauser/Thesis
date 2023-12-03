import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys

def interactive_plot(folder, timestep, alpha=10):

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    
    # Load the data
    P = scipy.io.loadmat(folder + f'/t{timestep}.mat');                    
    x=P['x']
    mask = np.load(folder + '/p_mask.npy')                                             

    # Create scatter object and fill in the data
    scatter1 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter1.set_data(x[mask==0] , edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)

    scatter2 = visuals.Markers(scaling=True, alpha=alpha, spherical=True)
    scatter2.set_data(x[mask==1] , edge_width=0, face_color=(1, 1, 1, .5), size=2.5)

    # Add the scatter object to the view
    view.add(scatter1)
    view.add(scatter2)

    # We want to fly around
    view.camera = 'fly'

    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()