from setuptools import setup, find_packages
# from pip.req import parse_requirements

# reqs = parse_requirements('requirements.txt')

setup(
    name="generic_pipelines",
    description='Utility staff for pytorch pipelining',
    version="0.1",
    # install_requires=reqs,
    # dependency_links = ['http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl'],
    packages=find_packages(exclude=('tests',)),
)
