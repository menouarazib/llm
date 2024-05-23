# Efficient Fine-Tuning of Open Source Language Learning Models (LLMs) Repository

<h1 align="center">
<img src="https://raw.githubusercontent.com/menouarazib/llm/b1bcf64094ae6228f5269030e469f8ea47c97945/finetuning_llms.jpg" width="800">
</h1><br>

This repository contains multiple Jupyter notebooks used for fine-tuning or RAG various Language Learning Models (LLMs). The models include but are not limited to, Falcon7B, Phi-2, and Zephyr7B.

Each notebook provides a detailed walkthrough of the fine-tuning or RAG process, including data preprocessing, model training, and evaluation. They serve as comprehensive guides for those interested in understanding and applying LLMs in their projects.

The fine-tuning techniques used in this repository include:

- [LoRA Adapter](https://huggingface.co/papers/2305.14314): A technique that allows for efficient fine-tuning and parameter sharing across multiple tasks.
- [Flash Attention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention): A novel attention mechanism that reduces the complexity of self-attention.
- [Quantization](https://huggingface.co/docs/transformers/main/en/quantization): A technique to reduce the memory footprint and improve the computational efficiency of the model.
- [Train on Completions Only using DataCollatorForCompletionOnlyLM](https://huggingface.co/transformers/main_classes/data_collator.html): A specific data collator for language model pretraining.
- [Using trl and the SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

Stay tuned for updates as we continue to add more models and improve our fine-tuning and RAG techniques.

**You can access my Hugging Face account [here](https://huggingface.co/Menouar).**
