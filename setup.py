from setuptools import setup, find_packages

setup(
    name='cltv_pareto_nbd_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'lifetimes',
        'dash',
        'plotly',
        'pytest',
    ],
    entry_points={
        'console_scripts': [
            'run_cltv_analysis=src.cltv_calculation:main',
        ],
    },
)