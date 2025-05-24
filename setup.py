from setuptools import setup,find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns a list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements
setup(
    name = 'mlproj',
    version = '0.0.1',
    author = 'Divyansh',
    author_email = 'divyanshgupta0804@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages = find_packages(),
    description = 'A package for ML projects',
)