from setuptools import find_packages,setup
from typing import List


hYPEN_E_DOT= '-e .'

def get_requriments(file_path:str)-> List[str]:
    """
    This function return the list of requriments
    """ 
    requriments = []

    with open(file_path ,'r') as file:
        requriments = file.readline()
        requriments = [req.replace('\n','') for req in requriments]
        if hYPEN_E_DOT in requriments:
            requriments.remove(hYPEN_E_DOT)

    return requriments

setup(
    name="mlprojects",
    version='0.0.1',
    author="Onkar korale",
    author_email="onkarkorale03@gmail.com",
    packages=find_packages(),
    install_requires = get_requriments('requirements.txt')
)