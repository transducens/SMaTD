#!/usr/bin/env python

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install

def reqs_from_file(src):
    requirements = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("-r"):
                requirements.append(line)
            else:
                add_src = line.split(' ')[1]
                add_req = reqs_from_file(add_src)
                requirements.extend(add_req)
    return requirements

#def post():
#    import nltk
#
#    try:
#        nltk.tokenize.word_tokenize("puc")
#    except LookupError:
#        nltk.download("punkt")
#
#class post_develop(develop):
#    def run(self):
#        develop.run(self)
#
#        post()
#
#class post_install(install):
#    def run(self):
#        install.run(self)
#
#        post()

if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()

    requirements = reqs_from_file("requirements.txt")

    setuptools.setup(
        name="MTDetect",
        version="1.0",
        install_requires=requirements,
        #license="GNU General Public License v3.0",
        author="Cristian García-Romero",
        author_email="cgr71ii@gmail.com",
        #maintainer=,
        #maintainer_email,
        description="Detection of machine translation",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/transducens/mtdetect",
        packages=["mtdetect", "mtdetect.utils"],
        #classifiers=[],
        #project_urls={},
        #package_data={  # Not available in the built package but just when building binaries!
        #    "mtdetect": [
        #    ]
        #},
        entry_points={
            "console_scripts": [
                "mtdetect = mtdetect.cli:main",
            ]
        },
        #cmdclass={
        #    "install": post_install,
        #    "develop": post_develop,
        #},
        )
