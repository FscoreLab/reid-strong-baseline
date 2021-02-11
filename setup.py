from setuptools import find_packages, setup

setup(name="reid_strong_baseline",
      version='0.0.2',
      description='reid spizzhennaya repo',
      url="https://github.com/FscoreLab/reid-strong-baseline",
      license="",
      packages=find_packages(),
      scripts=[
          'tools/train.py',
          'tools/test.py'
      ]
)
