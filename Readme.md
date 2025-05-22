# Client Resistance

Welcome to the official repository for our paper: "RECAP: Resistance Capture in Online Mental Health Counseling with Large Language Models."

This repository contains:

🧠 The resistance behavior framework proposed in our paper

📂 Sample data from the collected dataset

🧑‍💻 Training code for resistance detection

🤖 Our fine-tuned best-performing models 




<div align="center">
<img src="figures/intro.png" alt="Example" width="400"/>
</div>



## PsyFIRE Taxonomy

Interpersonal resistance refers to client behaviors that hinder or counteract the counselor’s efforts during the therapeutic process. Our proposed taxonomy, PsyFIRE, introduces a comprehensive framework comprising four overarching categories and thirteen distinct resistance behaviors. This structure enables a nuanced understanding and systematic capture of client resistance in text-based counseling interactions.


## Data

To facilitate researchers in understanding the format and patterns of our collected data, we provide sample data in the `data/` directory.
The full dataset will be made available upon request and subject to a data-sharing agreement. Access is limited to research teams for academic and non-commercial use only. To request access, please submit an issue on this repository.

## Model Training

You can perform training using the following configuration files: 
`examples/train_lora/llama3_full_sft_resistance.yaml`, 
`examples/train_lora/llama3_lora_sft_resistance.yaml`, 
and `examples/train_lora/llama3_full_dpo_resistance.yaml`. 
Each script corresponds to a specific training strategy.



## Model Checkpoint

If you're interested in our best-performing model, feel free to submit an issue on this repository to request access.





