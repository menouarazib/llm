# Efficient Fine-Tuning of Open Source Language Learning Models (LLMs) Repository

<h1 align="center">
<img src="https://raw.githubusercontent.com/menouarazib/llm/7d1aeb95fa034ddbbaecfe988a8988331c2770f3/finetuning_llms.jpg" width="800">
</h1><br>

This repository contains multiple Jupyter notebooks used for fine-tuning open source Language Learning Models (LLMs). The models include but are not limited to, Gemma, Falcon7B, Phi-2, and Zephyr7B.

Each notebook provides a detailed walkthrough of the fine-tuning, including data preprocessing, and model training. They serve as comprehensive guides for those interested in understanding and applying LLMs in their projects.

Here are the major steps involved in each notebook:
- **Set Up the Development Environment:** Prepare the necessary software and libraries for the project.
- **Load and Prepare the Dataset:** Import the dataset and preprocess it for the model.
- **Load the Base Model:** Load the model for fine-tuning.
- **Fine-Tune the LLM:** Adjust the LLM model parameters on our dataset.
- **Push the Fine-Tuned Model to the Hugging Face Hub.**

## The Efficient Techniques Used Are:
### Quantization
Quantization is the process of constraining an input from a continuous (large set of values) to a discrete set.

Mathematically, a linear quantization is an affine mapping of integers to floating points, we can write the equation as:

$$ r = (q - Z) \times S $$

Where:
- $r:$ Floating-point
- $q:$ Integer
- $Z:$ Zero point
- $S:$ Scale
- $r_{max}:$ maximum float in Tensor
- $r_{min}:$ minimum float in Tensor
- $q_{max}:$ maximum integer ($N$ bits, $2^{N-1} -1$)
- $q_{min}:$ minimum integer ($N$ bits, $-2^{N-1}$)


<h1 align="center">
<img src="https://raw.githubusercontent.com/menouarazib/llm/bb58c13bf0725015feba8efabc15ef17bfec105f/quantization.PNG" width="800">
  
[TinyML and Efficient Deep Learning Computing, Song Han]
</h1><be>

$$ S = \dfrac{r_{max} - r_{min}}{q_{max} - q_{min}}$$

$$ Z = q_{min} - \dfrac{r_{min}}{S} $$

$$ Z = round(q_{min} - \dfrac{r_{min}}{S}) $$

The basic concept of **neural network quantization** is converting the weights and activations of a neural network into a limited discrete set of numbers.
The most well-known quantization methods are:
- **Post-Training Quantization (PTQ)**: The quantization is done after the model is trained.
- **Quantization Aware Training (QAT)**: The quantization is applied to the model, and then it is retrained or fine-tuned.

<h1 align="center">
<img src="https://raw.githubusercontent.com/menouarazib/llm/6775596f83879183c2079f9215352a434b11d98e/PQT%20-%20QAT.png" width="800">
  
[Olivia Weng]
</h1><be>


The fine-tuning techniques used in this repository include:

- [LoRA Adapter](https://huggingface.co/papers/2305.14314): A technique that allows for efficient fine-tuning and parameter sharing across multiple tasks.
- [Flash Attention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention): A novel attention mechanism that reduces the complexity of self-attention.
- [Quantization](https://huggingface.co/docs/transformers/main/en/quantization): A technique to reduce the memory footprint and improve the computational efficiency of the model.
- [Train on Completions Only using DataCollatorForCompletionOnlyLM](https://huggingface.co/transformers/main_classes/data_collator.html): A specific data collator for language model pretraining.
- [Using trl and the SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

Stay tuned for updates as we continue to add more models and improve our fine-tuning and RAG techniques.

**You can access my Hugging Face account [here](https://huggingface.co/Menouar).**
