import os
import numpy as np
import jax.numpy as jnp
from jax import nn

def standardize(train_images, test_images, num_channels=1):
    mean = train_images.reshape(-1, num_channels).mean(0)
    std = train_images.reshape(-1, num_channels).std(0)
    return (train_images - mean) / std, (test_images - mean) / std

def load_mnist():
    import gzip
    from urllib import request

    def download_file(url, filename):
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            request.urlretrieve(url, filename)

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    urls = {
        'train_images': base_url + 'train-images-idx3-ubyte.gz',
        'train_labels': base_url + 'train-labels-idx1-ubyte.gz',
        'test_images': base_url + 't10k-images-idx3-ubyte.gz',
        'test_labels': base_url + 't10k-labels-idx1-ubyte.gz',
    }
    os.makedirs('data', exist_ok=True)
    files = {}
    for k, url in urls.items():
        filename = os.path.join('data', f'mnist_{k}.gz')
        download_file(url, filename)
        files[k] = filename

    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    train, test = standardize(
        jnp.array(load_images(files['train_images'])[..., None], dtype=jnp.float32),
        jnp.array(load_images(files['test_images'])[..., None], dtype=jnp.float32))
    train_ds = {'image': train, 'label': jnp.array(load_labels(files['train_labels']), dtype=jnp.int32)}
    test_ds = {'image': test, 'label': jnp.array(load_labels(files['test_labels']), dtype=jnp.int32)}
    return train_ds, test_ds