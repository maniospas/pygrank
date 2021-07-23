import setuptools

# Developer self-reminder for uploading in pypi:
# - install: wheel, twine
# - build  : python setup.py bdist_wheel
# - deploy : twine upload dist/*

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name='pygrank',
    version='0.1.17',
    author="Emmanouil Krasanakis",
    author_email="maniospas@hotmail.com",
    description="Recommendation algorithms for large graphs (compatible with networkx)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MKLab-ITI/pygrank",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent",
     ],
    install_requires=[
          'tqdm', 'sklearn', 'scipy', 'numpy', 'networkx',
      ],
 )