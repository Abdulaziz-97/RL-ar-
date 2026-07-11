from setuptools import setup, find_packages

setup(
    name="rlvr-pipeline",
    version="0.1.0",
    packages=find_packages(where="src") + find_packages(where="."),
    package_dir={"rlvr_pipeline": "src/rlvr_pipeline", "rlvr": "rlvr"},
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "trl>=0.9",
        "peft>=0.10",
        "bitsandbytes>=0.43",
        "datasets>=2.14",
        "accelerate>=0.28",
        "pyyaml",
    ],
    python_requires=">=3.10",
)
