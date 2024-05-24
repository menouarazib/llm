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

Mathematically, for linear quantization from floating points to integers, we can write the equation as:

$$ r = (q - Z) * S $$

Where:
- r: Floating-point
- q: Integer
- Z: Zero point
- S: Scale

<h1 align="center">
<img src="https://raw.githubusercontent.com/menouarazib/llm/7d1aeb95fa034ddbbaecfe988a8988331c2770f3/finetuning_llms.jpg" width="800">
</h1><br>


The quantization process involves loading the base model with lower precision. For example, if the base model weights are stored in 32-bit floating points and we decide to quantize them to 16-bit floating points, then the model size is divided by two. This makes it easier to store and reduces its memory usage. It can also speed up inference because it takes less time to perform calculations with fewer bits.





The fine-tuning techniques used in this repository include:

- [LoRA Adapter](https://huggingface.co/papers/2305.14314): A technique that allows for efficient fine-tuning and parameter sharing across multiple tasks.
- [Flash Attention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention): A novel attention mechanism that reduces the complexity of self-attention.
- [Quantization](https://huggingface.co/docs/transformers/main/en/quantization): A technique to reduce the memory footprint and improve the computational efficiency of the model.
- [Train on Completions Only using DataCollatorForCompletionOnlyLM](https://huggingface.co/transformers/main_classes/data_collator.html): A specific data collator for language model pretraining.
- [Using trl and the SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

Stay tuned for updates as we continue to add more models and improve our fine-tuning and RAG techniques.

**You can access my Hugging Face account [here](https://huggingface.co/Menouar).**
