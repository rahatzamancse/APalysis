from setuptools import setup, find_packages

setup(
    name='ChannelExplorer',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A tool for exploring channels with support for TensorFlow and PyTorch.',
    long_description=open('../README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ChannelExplorer',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Common dependencies here, if any
        'numpy',  # Example common dependency
        # other common dependencies
    ],
    extras_require={
        'tensorflow': ['tensorflow>=2.0'],
        'pytorch': ['torch', 'torchvision'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    python_requires='>=3.7',
)
