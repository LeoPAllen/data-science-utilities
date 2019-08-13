from setuptools import setup

setup(name='datascienceutils',
      version='0.1',
      description='Utility Functions For Data Science',
      url='https://github.com/LeoPAllen/datascienceutils',
      author='Leo Allen',
      author_email='leopallen@gmail.com',
      license='MIT',
      packages=['datascienceutils'],
      install_requires=[
          'pandas',
          'sklearn',
      ],
      zip_safe=False)
