import setuptools


VERSION = "0.0.1"
DEPS = [
  'numpy>=1.12.1',
  'nibabel'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sam",
    version=VERSION,
    author="Valerie Hayot-Sasson, Yongping Gao, Tristan Glatard",
    author_email="v_hayots@encs.concordia.ca",
    description="Tool to split and merge 3D neuroimaging (NIfTI) data",
    license="GPL3.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=['tests']),
    include_package_data=True,
    test_suite="pytest",
    tests_require=["pytest"],
    setup_requires=DEPS,
    install_requires=DEPS,
    python_requires=">=2.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux"
    ]
)
