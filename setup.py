from setuptools import setup, find_packages

setup(
    name="more_metrics",  
    version="0.11.0", 
    author="Lambert T Leong",  
    author_email="lamberttleong@gmail.com",  
    description="A Python package for computing additional classification metrics for machine learning or AI models. These metrics include and also known as AUC, NRI, and IDI",  
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/LambertLeong/AUC_NRI_IDI_python_functions", 
    license="GPL-3.0",  
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
	'matplotlib',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Minimum version requirement of the Python for your package
)

