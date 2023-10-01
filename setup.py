from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setup(
    name="DDChatbot",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
)


