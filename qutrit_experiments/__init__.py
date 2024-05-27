"""Main qutrit-experiments public functionality."""

import jax
jax.config.update('jax_enable_x64', True)
#  Uncomment if GPU is causing troubles
# jax.config.update('jax_platform_name', 'cpu')

__version__ = '1.0.0'
