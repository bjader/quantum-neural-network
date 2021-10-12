from setuptools import setup, find_packages

setup(
    name='qnn',
    version='0.1.2',
    author='Benjamin Jaderberg',
    author_email='benjamin.jaderberg@physics.ox.ac.uk',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/bjader/quantum-neural-network',
    license='LICENSE',
    description='For building quantum neural networks in Qiskit and integrating with PyTorch',
    long_description=open('README.md').read(),
    install_requires=[
        "qiskit"
    ],
)