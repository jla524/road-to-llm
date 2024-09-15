from setuptools import setup, find_packages

setup(
    name="road_to_llm",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["tinygrad", "numpy", "tiktoken", "transformers[torch]", "datasets", "evaluate", "scikit-learn"],
)
