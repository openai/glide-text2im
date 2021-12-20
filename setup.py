from setuptools import setup

setup(
    name="glide-text2im",
    packages=["glide_text2im"],
    install_requires=[
        "Pillow",
        "attrs",
        "torch",
        "filelock",
        "requests",
        "tqdm",
    ],
    author="OpenAI",
)
