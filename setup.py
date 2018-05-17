from setuptools import setup, find_packages
# To use consisten encodings
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

requirements = [
    'torch',
    'numpy',
    'opencv-python',
    'enum34;python_version<"3.4"'
]

setup(
    name='face_iqa',
    version='0.1.0',

    description="Detector face quality assessment from Python",
    long_description=long_description,

    # Package info
    packages=find_packages(exclude=('log', 'data', 'src', 'model')),

    install_requires=requirements,
    license='BSD',
    zip_safe=True,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',

        # Supported python versions
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
