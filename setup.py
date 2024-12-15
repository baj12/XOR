from setuptools import setup, find_packages

setup(
    name='xorProject',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'tensorflow',
        'keras',
        'deap',
        'PyYAML',
    ],
    entry_points={
        'console_scripts': [
            'xorProject=src.main:main',
        ],
    },
    author='Bernd Jagla',
    author_email='bernd.jagla@pasteur.fr',
    description='XOR problem solver using genetic algorithms and machine learning.',
    url='https://github.com/baj12/xorProject',
)
