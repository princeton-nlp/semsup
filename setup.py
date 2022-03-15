import setuptools

setuptools.setup(
    name="semsup",
    version="0.1.1",
    author="Austin Wang Hanjie, Ameet Deshpande",
    author_email="hjwang@cs.princeton.edu",
    description="Semantic Supervision",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'torch==1.9.1',
        'pytorch-lightning==1.4.9',
        'torchvision==0.10.1',
        'transformers==4.11.3',
        'datasets==1.13.2',
        'scikit-learn==1.0',
        'gensim==4.1.2',
        'jsonargparse==4.1.0',
        'wandb==0.12.9'
    ],
)