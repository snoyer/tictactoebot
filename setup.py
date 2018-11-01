import os, sys
import subprocess
import setuptools
from setuptools.command.install_lib import install_lib



def setup():
    opencv_deps = pkgconfig('opencv')
    setuptools.setup(
        name='tictactoebot',
        version='0.0.1',
        description='opencv tictactoe',
        url='http://github.com/snoyer/tictactoebot',
        author='snoyer',
        author_email='reach me on github',
        packages=['tictactoebot'],
        install_requires=[
            'opencv-python',
            'opencv-contrib-python',
            'requests',
        ],
        ext_modules=[
            setuptools.Extension(
                name='_tictactoe',
                sources=['opencv/pythonmodule.cpp', 'opencv/pyboostcvconverter/pyboost_cv3_converter.cpp'],
                include_dirs = ['opencv/pyboostcvconverter'] + opencv_deps['include_dirs'],
                libraries = ['boost_python3' if sys.version_info >= (3,0) else 'boost_python2'] + opencv_deps['libraries'],
                language='c++11',
                extra_compile_args=['-O3'],
            ),
        ],
    )



def pkgconfig(*packages, **kw):
    """pkg-config function from https://gist.github.com/abergmeier/9488990"""
    flag_map = {
        '-I': 'include_dirs',
        '-L': 'library_dirs',
        '-l': 'libraries',
    }

    env = os.environ.copy()
    for token in subprocess.check_output(['pkg-config', '--libs', '--cflags', ' '.join(packages)], env=env).decode().split():
        key = token[:2]
        try:
            arg = flag_map[key]
            value = token[2:]
        except KeyError:
            arg = 'extra_link_args'
            value = token
        kw.setdefault(arg, []).append(value)

    for key, value in kw.items(): # remove duplicated
        kw[key] = list(set(value))
    return kw


setup()
