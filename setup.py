from setuptools import find_packages, setup

setup(
    name="gajahmada-ai",
    version="2.0.0",
    description="GajahMada — Sovereign Decentralized Node & Web3 AI Architecture",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.103.0",
        "uvicorn>=0.23.2",
        "pydantic>=2.3.0",
        "cryptography>=41.0.3",
        "httpx>=0.25.0",
        "click>=8.1.7",
        "PyYAML>=6.0.1",
        "psutil>=5.9.5",
    ],
    extras_require={
        "ml": [
            "llama-cpp-python>=0.2.11",
            "sentence-transformers>=2.2.2",
            "faiss-cpu>=1.7.4",
            "numpy>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gajahmada=gajahmada_ai.cli.main:main",
        ],
    },
)
