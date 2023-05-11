from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
  name = 'vision_xformer',
  packages = find_packages(exclude=['examples']),
  version = '0.2.0',
  license='MIT',
  description = 'Vision Xformers',
  long_description=long_description,
  long_description_content_type = 'text/markdown',
  author = 'Pranav Jeevan',
  author_email = 'pranav13phoenix@gmail.com',
  url = 'https://github.com/pranavphoenix/VisionXformer',
  keywords = [
    'artificial intelligence',
    'training',
    'optimizer',
    'machine learning',
    'attention',
    'transformers',
    'computer vision'
  ],
  install_requires=[
    'torch',
    'torchvision',
    'numpy',
    'einops',
    'performer-pytorch',
    'linformer'
  ],
  
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)