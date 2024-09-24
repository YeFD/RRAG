import random, os
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from typing import Any, Dict, List, Optional, Tuple

class RRAGLlamaConfig(PretrainedConfig):
    model_type = "rragllama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_name_or_path='',
        load_in_8bit=True,
        input_dim=3,
        hidden_size=4096,
        unk_token='<unk>',
        unk_token_id=0,
        similarity_token='<R>',
        freeze_llm=False,
        num_k=10,
        d_model=256,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.load_in_8bit = load_in_8bit
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.unk_token = unk_token
        self.unk_token_id = unk_token_id
        self.similarity_token = similarity_token
        self.freeze_llm = freeze_llm
        self.num_k = num_k
        self.d_model = d_model
        super().__init__(
            **kwargs,
        )

class RFormer(nn.Module):
    def __init__(self, input_dim, num_k, d_model=256, n_head=4, num_layers=1, dropout=0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_k = num_k
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.input_layer = nn.Linear(in_features=input_dim, out_features=d_model)
        self.position_embeddings = nn.Embedding(num_k, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier_layer = nn.Linear(in_features=d_model, out_features=1)
        self.classification_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, x, label=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.input_layer(x)
        position_ids = torch.arange(self.num_k, device=x.device).expand(x.shape[0], self.num_k)
        pe = self.position_embeddings(position_ids)
        x = x + pe
        x = self.attention_layer(x.permute(1, 0, 2)).permute(1, 0, 2)
        y = self.classifier_layer(x)
        if label is not None:
            label = label.to(y.dtype)
        loss = self.classification_loss(y, label) if self.training else None
        return x, loss

class RRAGLlamaForCausalLM(PreTrainedModel):
    config_class = RRAGLlamaConfig
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
            print('freeze_llm')
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        self.r_former = RFormer(input_dim=config.input_dim, num_k=config.num_k, d_model=config.d_model).to(self.llama_model.device)
        self.llama_proj = nn.Linear(config.d_model, config.hidden_size, device=self.llama_model.device)

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

    def encode_retrieval_data(
        self, 
        embeds: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        ):
        if embeds.dim() <= 2 and self.config.input_dim == 1:
            embeds = embeds.unsqueeze(-1)
        if label is not None and label.dim() <= 2:
            label = label.unsqueeze(-1)
        logits, loss = self.r_former(embeds, label)
        inject_embeds = self.llama_proj(logits)
        return inject_embeds, loss

    def encode_inputs(self, 
        input_ids: torch.Tensor, 
        embeds: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ):
        embed_tokens = self.get_input_embeddings()
        input_embeds = embed_tokens(input_ids)
        if embeds is not None:
            inject_embeds, loss = self.encode_retrieval_data(embeds, label)
            unk_token_id = self.config.unk_token_id
            if input_embeds.dim() == 2:
                raise ValueError('dim error')
            elif input_embeds.dim() == 3:
                updated_input_embeds = input_embeds.clone()
                replace_idx = torch.nonzero(input_ids==unk_token_id).squeeze()
                inject_embeds = inject_embeds.reshape([-1, inject_embeds.shape[-1]])
                updated_input_embeds[replace_idx[:, 0], replace_idx[:, 1]] = inject_embeds.to(input_embeds.dtype)
                return updated_input_embeds.contiguous(), loss
        else:
            return input_embeds, None

    
    def forward(
        self,
        input_ids: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if embeds is None:
            raise ValueError('embeds is None')
        if label is None:
            raise ValueError('label is None')
        inputs_embeds, loss = self.encode_inputs(input_ids, embeds, label)
        if 'inputs_embeds' in kwargs:
            _ = kwargs.pop('inputs_embeds')
        outputs = self.llama_model(inputs_embeds=inputs_embeds, **kwargs)
        if self.training:
            outputs.loss += loss
        return outputs

    def generate(
        self,
        inputs: torch.Tensor,
        **kwargs
    ):
        if 'embeds' not in inputs.keys():
            raise ValueError('embeds is None')
        inputs_embeds, loss = self.encode_inputs(inputs['input_ids'], inputs['embeds'])
        if 'inputs_embeds' in kwargs:
            _ = kwargs.pop('inputs_embeds')
        outputs = self.llama_model.generate(inputs_embeds=inputs_embeds, **kwargs)
        return outputs

    def save_model(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        model_to_save = self.llama_model.module if hasattr(self.llama_model, 'module') else self.llama_model
        if not self.config.freeze_llm:
            model_to_save.save_pretrained(save_directory)
        model_dict = {
            'r_former': self.r_former.state_dict(),
            'llama_proj': self.llama_proj.state_dict()
        }
        torch.save(model_dict, os.path.join(save_directory, 'RRAGLlama_pytorch_model.bin'))

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            raise ValueError("Configuration must be provided with `config` argument.")
        
        model = cls(config)

        llm_path = config.model_name_or_path if config.freeze_llm else pretrained_model_path
        print(f'Load LLM params from: {llm_path}')
        llama_model_class = AutoModelForCausalLM.from_pretrained
        llama_model = llama_model_class(llm_path, device_map="auto", load_in_8bit=config.load_in_8bit)
        model.llama_model = llama_model
        
        model_path = os.path.join(pretrained_model_path, 'RRAGLlama_pytorch_model.bin')
        other_model_dict = torch.load(model_path)
        model.r_former.load_state_dict(other_model_dict['r_former'])
        model.llama_proj.load_state_dict(other_model_dict['llama_proj'])
        return model
