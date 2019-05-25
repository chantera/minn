from setuptools import find_packages, setup


setup(
    name='minn',
    version='0.1.0',
    author='Hiroki Teranishi',
    author_email='teranishihiroki@gmail.com',
    description='A Minimum Neural Network Toolkit',
    url='https://github.com/chantera/minn',
    license='MIT License',
    install_requires=['numpy>=1.11.0'],
    packages=find_packages(),
)
