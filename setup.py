import os
import sys
import setuptools


setuptools.setup(
    name='montecarlo',
    version=0.1,
    author="The Montecarlo Team",
    author_email="montecarlo@amazon.com",
    description="montecarlo",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/awslabs/montecarlo_tf",
    packages=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    #pinning aioboto3 version as aiobot3 is pinning versions
    # https://github.com/aio-libs/aiobotocore/issues/718
    install_requires = [ 'boto3==1.9.91','numpy'],
        setup_requires=["pytest-runner"],
        #tests_require=tests_packages,
        python_requires='>=3.6'
    )

