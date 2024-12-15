from setuptools import setup, find_packages

setup(
    name='your_project',
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
            'your_project=src.main:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of your project',
    url='https://github.com/yourusername/your_project',
)
