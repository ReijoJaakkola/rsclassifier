from setuptools import setup

setup(name = 'rsclassifier',
      version = '2.0.0',
      author = 'Reijo Jaakkola',
      author_email = 'jaakkolareijo@hotmail.com',
      description = 'Package for training rule set classifiers for tabular data.',
      packages = [
          'rsclassifier',
          'discretization'
      ]
)