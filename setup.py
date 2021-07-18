from pathlib import Path

from setuptools import find_packages, setup

readme_file = Path(__file__).parent / 'README.md'
if readme_file.exists():
    with readme_file.open() as f:
        long_description = f.read()
else:
    long_description = ''

setup(
    name='digit-recognizer',
    version='1.0.0',
    description='Digit Recognizer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/f4str/digit-recognizer',
    license='MIT',
    author='Farhan Ahmed',
    keywords='',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.6',
    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=['Pillow', 'numpy', 'torch', 'torchvision', 'tqdm'],
    extras_require={'dev': ['pytest', 'tox']},
)
