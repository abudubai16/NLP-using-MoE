# Switch Transformer with Mixture of Experts (MoE)

[Paper Link](https://arxiv.org/abs/2101.03961)

This repository contains my implementation of the **Switch Transformer** architecture using **PyTorch**. The Switch Transformer is a variant of the Transformer model that leverages **Mixture of Experts (MoE)** to achieve efficient scaling by only activating a small subset of parameters (experts) for each input, thus reducing computational cost. I have extended PyTorch's `TransformerEncoderLayer` by rewriting the MLP part of the layer to utilize the MoE approach. Additionally, I implemented a custom loss function to ensure an even distribution of tokens across the experts, preventing any single expert from being overloaded.

## Overview

The Switch Transformer model in this project is designed for natural language processing (NLP) tasks and aims to improve model efficiency by using MoE, which dynamically selects a subset of **experts** to process each input token. The model has been trained successfully and produces coherent outputs; however, I have observed that the model's performance starts to degrade after a certain number of epochs, indicating some robustness challenges with the MoE architecture.

## Mixture of Experts (MoE)

### What is Mixture of Experts?

A **Mixture of Experts (MoE)** is a model that consists of multiple independent networks, called **experts**, where only a subset of experts are active for a given input. Instead of using all parameters for each input (as in traditional models), the MoE approach selects a small number of experts to process the input, leading to more efficient use of computational resources.

In the Switch Transformer:
- Each input token is routed to one or a few experts using a **router** based on the input's characteristics.
- The MLP layer inside each Transformer layer is replaced with an MoE, where each expert is a feed-forward network.

### Advantages of MoE

- **Scalability**: MoE allows scaling models to have billions or even trillions of parameters without increasing the computational cost linearly, as only a small number of experts are active for each token.
- **Efficient Computation**: By activating fewer experts, MoE reduces the number of parameters involved in each forward pass, which leads to faster training and inference times.
- **Capacity for Specialization**: Since experts can learn specialized skills for particular types of inputs, MoE models can capture more specific patterns in the data.

### Loss Function for Expert Balance

To prevent certain experts from being overloaded with too many tokens while others remain underused, I implemented a custom loss function to ensure an even distribution of tokens across experts.

## Challenges and Observations

During the training process, I observed the following:
- The model produces coherent and meaningful outputs after a few epochs.
- **Degradation**: After a certain number of epochs, the model’s performance (specifically the MoE component) starts to degrade. This happens much earlier than expected for traditional NLP models, indicating potential issues with robustness or token distribution across experts.
- **Token Distribution**: Despite the custom loss function, balancing the token distribution across experts can be challenging, especially when the dataset becomes highly specialized or imbalanced.

## Future Direction

While this project focuses on the Switch Transformer architecture, I plan to explore other Mixture of Experts architectures to further improve performance and robustness. These could include more complex expert routing mechanisms and diverse MoE designs beyond Switch Transformers to address current limitations and achieve better results.

---

This project demonstrates the Switch Transformer model with Mixture of Experts and serves as a foundation for exploring various MoE architectures.
