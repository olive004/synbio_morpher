import jax
import jax.numpy as jnp


jax.config.update('jax_platform_name', 'gpu')


def test_vmap(m):
    return jnp.sin(m)


jax.vmap(test_vmap)(jnp.ones(4))