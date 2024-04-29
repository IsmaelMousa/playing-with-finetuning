# Fine-Tuning a Pre-trained Model

## Overview
This repository serves as a practice on how to fine-tune a pre-trained transformers model from the Hugging Face.

## Points
- Practice on how to prepare the foundations of the training environment, such as the **tokenizer**, **model**, **dataset**, and **hyperparameters**.
- Practice on how to fine-tune a model by using the `Trainer` API.
- Practice on how to evaluate the performance of the model after finishing the training process, by using the **evaluate** library.

## Usage
1. Clone this repository to your local machine:
```zsh
git clone git@github.com:IsmaelMousa/playing-with-finetuning.git
```
2. Navigate to the **playing-with-finetuning** directory:
```zsh
cd playing-with-finetuning
```
3. Setup virtual environment:
```zsh
python3 -m venv .venv
```
4. Activate the virtual environment:

```zsh
source .venv/bin/activate
```
5. Install the required dependencies:

```zsh
pip install -r requirements.txt
```