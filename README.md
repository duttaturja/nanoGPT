# nanoGPT: Building a Minimal GPT from Scratch

This repository is a personal project inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/build-nanoGPT) and his instructional video on constructing GPT models from scratch. The aim is to delve deep into the architecture and training processes of Generative Pre-trained Transformer (GPT) models, providing a hands-on approach to understanding their inner workings. The implementation shown in his video is in Separated folder named "Andrej Karpathy's Implementation".

My Implementation is an advanced, modular implementation of nanoGPT with:
- **Custom Transformer Architecture** 
- **Federated Learning via Flower**
- **Local Training / Text Generation / CLI Interface**
- **Designed for extensibility and real-world FL simulations**

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)
- [Contribution](#contributing)

##  Introduction

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

---

## Dataset

The project uses the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

---

## Papers

This project is grounded in seminal papers:

- **Attention Is All You Need** (Vaswani et al., 2017): Introduced the Transformer architecture with multi-head self-attention, positional encoding, and the encoder-decoder structure leveraged in `transformer.py`.

- **Language Models are Unsupervised Multitask Learners (GPT-2)** (Radford et al., 2019): Presented the decoder-only GPT-2 model, causal self-attention, layer normalization, and the training strategies (AdamW optimizer, cross-entropy loss) implemented in `model.py` and `GPT.py`.

- **Language Models are Few-Shot Learners (GPT-3)** (Brown et al., 2020): Scaled up GPT-2 to billions of parameters, demonstrating zero- and few-shot learning. This informed our generation procedure and sampling temperature strategies.


## Usage

Run all experiments using the `GPT.py` CLI entrypoint:

### 1. Local Training
Train the model locally using your selected architecture:
```bash
python train.py
```

### 2. Text Generation
Generate text from a trained checkpoint:
```bash
python GPT.py
```
### 3. Federated Learning (Flower)
#### Start simulation:
```bash
python federated.py --iid
```
#### Start Server:
```bash
python -m Federated_Learning.server
```

#### Start Clients (in separate terminals):
```bash
python -m Federated_Learning.client --cid=0
python -m Federated_Learning.client --cid=1
```
> Each client trains on its own partition of Tiny Shakespeare and syncs with the central server.

---
## Project Structure

```
nanoGPT/
├── .gitignore                # Ignore unnecessary files in version 
├── Andrej Karpathy's Implementation
├── ├── GPT.py                # Core GPT model implementation
├── ├── play.ipynb            # Jupyter Notebook for 
├── Dataset
├── ├── me.txt                # Dataset
├── ├── tiny_shakespeare.txt  # Dataset
├── Federated_Learning
├── nanoGPT
├── ├── data_loader.py        # Dataset preprocessing and loader
├── ├── config.py             # Configuration file for GPT
├── ├── model.py              # GPT model
├── ├── transformer.py        # Attention Transformer implementation
├── ├── utils.py              # Data helpers, save/load, text gen
├── Papers
├── ├── Attention Is All You Need.pdf
├── ├── Language Models are Few-Shot Learners.pdf
├── ├── Language Models Are Unsupervised Multitask Learners.pdf
├── nanoGPT.py                # Main entrypoint 
├── train.py                  # Local training script
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # You're here!
```

---

## Notes
- Checkpointing is handled in `utils.py` as `ckpt.pt`.
- You can choose between `gpt` and `transformer` models in CLI.
- Ports must be open when running FL on localhost.
- Easily extend this for other text datasets, FL strategies (FedAvg, FedProx), or mixed-expert models.

---

## References

This is a direct implementation of andrej karpathys Let's reproduce GPT-2 (124M). I have added My own implementation as well with my own transformer and federated learning to train the LLM. Massive shoutout to Mr.Karpathy for the education content he provides. All important links of Mr.Karpathy required for this repository are given below:

- [build-nanoGPT by Andrej Karpathy](https://github.com/karpathy/build-nanoGPT)
- [Let's reproduce GPT-2 (124M).](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Andrej Karpathy](https://github.com/karpathy)

## Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please follow these steps:

1. **Fork the repository** and create your branch from `main`.
2. **Create an issue** to discuss what you would like to change.
3. **Make your changes** with clear, descriptive commit messages.
4. **Test your code** to ensure compatibility and stability.
5. **Open a pull request** describing your changes and the motivation behind them.

All contributions should follow the existing code style and include relevant documentation and tests where appropriate.

Thank you for helping improve this project!
<p align=center>
~ <a href="https://github.com/duttaturja">Turja Dutta </a>
</p>

---
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
