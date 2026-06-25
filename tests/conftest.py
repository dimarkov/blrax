import jax

# Enable 64-bit float precision for tests to avoid numerical precision issues
jax.config.update("jax_enable_x64", True)
