import jax.numpy as jnp
from jax import random

def get_random_points_on_sphere(shape : int, key_int : int) -> jnp.ndarray:
    key = random.PRNGKey(key_int)
    theta = 2 * jnp.pi * random.uniform(key, (shape,))

    key = random.PRNGKey(key_int+2)
    phi = jnp.arccos(1 - 2 * random.uniform(key, (shape,)))

    x = jnp.sin(phi) * jnp.cos(theta)
    y = jnp.sin(phi) * jnp.sin(theta)
    z = jnp.cos(phi)
    return jnp.array([x, y, z]).T