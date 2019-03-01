from setuptools import setup, find_packages

setup(
    name='pillmatch',
    version='0.1',
    description='A library for matching pill images',
    long_description="",
    long_description_content_type="text/markdown",
    url='https://github.com/electricbrainio/eb-pill-match',
    author='Bradley Arsenault',
    author_email='brad@electricbrain.io',
    license='Proprietary',
    python_requires='>=3',
    packages=find_packages(),
    package_data={
        'pillmatch': [
        ]
    },
    install_requires=[
        "hypermax",
        "gunicorn",
        'lightgbm',
        "matplotlib",
        "Flask",
        # "tensorflow-gpu",
        "scikit-learn",
        "keras",
        'azure-storage',
        'exifread'
    ],
    classifiers=[
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    platforms=['Linux', 'OS-X'],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'pm_api_server = pillmatch.bin.api_server:main',
            'pm_test_model = pillmatch.bin.test_model:main',
            'pm_train_model = pillmatch.bin.train_model:main',
            'pm_prepare_database_vectors = pillmatch.bin.prepare_database_vectors:main',
            'pm_generate_example_pill_images = pillmatch.bin.generate_example_pill_images:main'
        ]
    }
)