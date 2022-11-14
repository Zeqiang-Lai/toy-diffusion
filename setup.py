from setuptools import setup, find_packages

setup(
  name = 'toy-diffusion',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  author = 'Zeqiang Lai',
  install_requires=[
    'accelerate',
    'einops',
    'ema-pytorch',
    'pillow',
    'torch',
    'torchvision',
    'tqdm'
  ],
)
