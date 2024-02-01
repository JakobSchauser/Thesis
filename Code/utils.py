import jax.numpy as jnp
from jax import random
import numpy as np

def get_random_points_on_sphere(shape : int, key_int : int) -> jnp.ndarray:
    key = random.PRNGKey(key_int)
    theta = 2 * jnp.pi * random.uniform(key, (shape,))

    key = random.PRNGKey(key_int+2)
    phi = jnp.arccos(1 - 2 * random.uniform(key, (shape,)))

    x = jnp.sin(phi) * jnp.cos(theta)
    y = jnp.sin(phi) * jnp.sin(theta)
    z = jnp.cos(phi)
    return jnp.array([x, y, z]).T



def voronoi_plot_2d(vor, ax=None, **kw):
    """
    Plot the given Voronoi diagram in 2-D

    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on
    show_points: bool, optional
        Add the Voronoi points to the plot.
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha: float, optional
        Specifies the line alpha for polygon boundaries

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    Voronoi

    Notes
    -----
    Requires Matplotlib.

    """
    from matplotlib.collections import LineCollection

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")
    point_alpha = kw.get('point_alpha', 1.0)

    if kw.get('show_points', True):
        ax.plot(vor.points[:,0], vor.points[:,1], '.', alpha=point_alpha)
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:,0], vor.vertices[:,1], 'o')

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)

        vert = vor.vertices[simplex]
        # find the length of the lines
        ls = np.linalg.norm(vert[0] - vert[1])
        if np.all(simplex >= 0) and np.all(ls < 2):
            finite_segments.append(vert)
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            infinite_segments.append([vor.vertices[i], far_point])

    ax.add_collection(LineCollection(finite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='solid'))
    # ax.add_collection(LineCollection(infinite_segments,
    #                                  colors=line_colors,
    #                                  lw=line_width,
    #                                  alpha=line_alpha,
    #                                  linestyle='dashed'))


    return ax.figure