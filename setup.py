from setuptools import setup

setup(
    name="mptb",
    version="0.0.1",
    description='BERT implementation of PyTorch',
    author="Toshihiko Aoki",
    url='https://github.com/to-aoki/my-pytorch-bert/',
    license='Apache License 2.0',
    install_requires=['torch>=0.4.1',
                    'sentencepiece',
                    'tqdm'
                    ],
    extras_require={
        "develop": ['tensorflow',
                    'scikit-learn',
                    'mecab-python3',
                    'pyknp'
                    'mojimoji'
                    ],
    }
)