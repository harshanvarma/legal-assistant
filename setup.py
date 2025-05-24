"""
Setup script for Multi-Agent Legal Research Companion
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="legal-research-companion",
    version="1.0.0",
    author="Legal Research AI Team",
    author_email="contact@legalresearch.ai",
    description="Multi-Agent Legal Research Companion for Indian Law Students",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/legal-research-companion/legal-research-companion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Professionals",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "groq>=0.8.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.2",
        "selenium>=4.15.0",
        "nltk>=3.8.1",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.8.5",
        "lxml>=4.9.3",
        "urllib3>=2.0.4",
        "fake-useragent>=1.4.0",
        "textblob>=0.17.1",
        "spacy>=3.7.2",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "sentence-transformers>=2.2.2",
        "pypdf2>=3.0.1",
        "python-docx>=0.8.11",
        "openpyxl>=3.1.2",
        "chromadb>=0.4.15",
        "langchain>=0.0.330",
        "langchain-community>=0.0.3",
        "networkx>=3.0",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "legal-research=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.json"],
    },
)