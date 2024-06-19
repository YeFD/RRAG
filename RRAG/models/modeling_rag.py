import random, os
import torch
import torch.nn as nn
from transformers import  LlamaForCausalLM, LlamaModel, PreTrainedModel, LlamaConfig, AutoModelForCausalLM, PretrainedConfig
from typing import Any, Dict, List, Optional, Tuple

class RAGLlamaConfig(PretrainedConfig):
    model_type = "ragllama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_name_or_path='',
        load_in_8bit=True,
        freeze_llm=True,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.load_in_8bit = load_in_8bit
        self.freeze_llm = freeze_llm
        super().__init__(
            **kwargs,
        )

class RAGLlamaForCausalLM(PreTrainedModel):
    config_class = RAGLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    def __init__(self, config):
        super().__init__(config)
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, 
            device_map="auto",
            load_in_8bit=config.load_in_8bit,
            )
        if config.freeze_llm:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

    def get_input_embeddings(self):
        return self.llama_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llama_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llama_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llama_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.llama_model.set_decoder(decoder)

    def get_decoder(self):
        return self.llama_model.get_decoder()

    def encode_inputs(self, 
        input_ids: torch.Tensor, 
    ):
        embed_tokens = self.get_input_embeddings()
        input_embeds = embed_tokens(input_ids)
        return input_embeds, None

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        inputs_embeds, loss = self.encode_inputs(input_ids)
        outputs = self.llama_model(inputs_embeds=inputs_embeds, **kwargs)
        return outputs

    def generate(
        self,
        inputs: torch.Tensor,
        **kwargs
    ):
        inputs_embeds, loss = self.encode_inputs(inputs['input_ids'])
        outputs = self.llama_model.generate(inputs_embeds=inputs_embeds, **kwargs)
        return outputs