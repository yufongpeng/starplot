from setuptools import setup, find_packages
import sys, os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.rst')).read()
NEWS = open(os.path.join(here, 'NEWS.txt')).read()


version = '0.1'

install_requires = [
    # List your project dependencies here.
    # For more details, see:
    # http://packages.python.org/distribute/setuptools.html#declaring-dependencies
    "scipy>=1.4",
    "seaborn>=0.10",
    "statsmodels>=0.11",
    "matplotlib>=3.2",
    "numpy>=1.18",
    "pandas>=1.0"
]


setup(name='starplot',
    version=version,
    description="Create barplots or boxplots with significant level annotations.",
    long_description=README + '\n\n' + NEWS,
    classifiers=[
      # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
    keywords='python numpy scipy statsmodels matplotlib seaborn t-test f-test mann-whitney non-parametric boxplots barplots significance-stars',
    author='yufongpeng',
    author_email='sciphypar@gmail.com',
    url='https://github.com/yufongpeng/starplot',
    license='',
    packages=find_packages('src'),
    package_dir = {'': 'src'},include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
        'console_scripts':
            ['starplot=starplot:main']
    }
)
