FROM nvcr.io/nvidia/jax:26.09-py3

# Keep the container's own JAX build (made for this GPU); pin it while adding project deps.
RUN pip list --format=freeze | grep -iE '^(jax|jaxlib|jax[-_]cuda[0-9]*[-_a-z]*)==' > /tmp/pins.txt \
 && cat /tmp/pins.txt \
 && pip install --no-cache-dir -c /tmp/pins.txt flax optax "imageio>=2.34" "imageio-ffmpeg>=0.5" pillow

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.09 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
