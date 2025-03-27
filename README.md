# nanoGPT: Building a Minimal GPT from Scratch

This repository is a personal project inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/build-nanoGPT) and his instructional video on constructing GPT models from scratch. The aim is to delve deep into the architecture and training processes of Generative Pre-trained Transformer (GPT) models, providing a hands-on approach to understanding their inner workings.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

## Introduction

GPT models have significantly advanced natural language processing by enabling machines to generate human-like text. This project endeavors to reconstruct a nano-scale GPT model from the ground up, offering insights into the architecture and training methodologies of GPT models. By following this guide, you'll gain a comprehensive understanding of how GPT models function and how to implement them.

## Features

- **Minimalistic Design:** Focuses on the essential components required to build and comprehend a GPT model.
- **Educational Walkthrough:** Each step is meticulously documented to facilitate learning and comprehension.
- **Customizable:** Designed to be easily extendable for experimentation with various architectures and datasets.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/duttaturja/nanoGPT.git
   cd nanoGPT
   ```

2. **Create a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installation, you can start training the nanoGPT model:

```bash
python GPT.py 
```


## Project Structure

```
nanoGPT/
├── .gitignore          # Ignore unnecessary files in version control
├── GPT.py              # Core GPT model implementation
├── input.txt           # Sample input text for testing
├── play.ipynb          # Jupyter Notebook for interactive experimentation
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── README.md           # Project documentation
```

## References

- [nanoGPT by Andrej Karpathy](https://github.com/karpathy/build-nanoGPT)
- [Let's reproduce GPT-2 (124M).](https://www.youtube.com/watch?v=l8pRSuU81PU)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
