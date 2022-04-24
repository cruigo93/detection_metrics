from setuptools import setup

setup(
    name='eval_detector',
    version='0.2.0',
    description='Python package to calculate recall and precision from bounding boxes',
    url='https://github.com/cruigo93/detection_metrics',
    author='Zhuldyzzhan Sagimbayev',
    author_email='cruigo93@gmail.com',
    packages=['eval_detector'],
    install_requires=['numpy',
                      'loguru'],
    scripts=['bin/eval_detector']


)