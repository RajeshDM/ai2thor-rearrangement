from setuptools import setup
from distutils.extension import Extension

setup(name='rearrangement_challenge',
      packages=['rearrange', 'baseline_configs'],
      version='0.1',
      description='Rearrangement Challenge',
      python_requires='>3.8',
      install_requires=[
          #'ai2thor==3.3.4',
          'ai2thor==5.0.0',
          'pomdp-py',
          'tqdm',
          'prettytable',
          'pytz',
          'pandas',
          'seaborn',
          'sciex==0.3',
      ],
      author='Rajesh Mangannavar',
      author_email='mangannr@oregonstate.edu')