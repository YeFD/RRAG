import torch
import numpy as np
import random, os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(torch.cuda.is_available())
def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
seed_it(42)


from tqdm import tqdm
import pickle
import argparse
from datetime import datetime
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType

from RRAG.dataset.load_nq import load_nq_dataset, get_nq_ans
from RRAG.dataset.load_hotpotqa import load_hotpotqa_dataset, get_hotpotqa_ans
from RRAG.dataset.load_musique import load_musique_dataset, get_musique_ans
from RRAG.models.modeling_rrag import RRAGLlamaForCausalLM, RRAGLlamaConfig
from RRAG.models.modeling_rag import RAGLlamaForCausalLM, RAGLlamaConfig
from RRAG.utils.trainer import RRAGTrainer
from RRAG.utils.metrics import evaluation_from_list

class RRAGRunner:
    RETRIEVAL_TOKEN = '<R>'
    UNK_TOKEN = '<unk>'
    UNK_TOKEN_ID = 0
    instruction_type = 'instruction'

    def __init__(
        self,
        dataset_name='nq_10',
        input_path='',
        max_prompt_length=4096,

        model_name='',
        load_in_8bit=False,
        save_model=False,
        output_dir='',

        use_rrag=True,
        input_dim=3,
        hidden_size=4096,
        RETRIEVAL_TOKEN='<R>',
        UNK_TOKEN='<unk>',
        UNK_TOKEN_ID = 0,

        num_k=10,
        use_lora=False,
        use_training=False,
        freeze_llm=True,
        load_from_pretrained=True,
        pretrained_model_name='',
        num_train_epochs=2,
        per_device_train_batch_size=2,

        use_evaluation=True,
        max_new_tokens=100,
        use_beam=False,
        beam_num=5,
        save_results=False,
        instruction_type='instruction',
    ):
        self.dataset_name = dataset_name # 
        self.input_path = input_path
        self.max_prompt_length = max_prompt_length

        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.save_model = save_model
        self.output_dir = output_dir

        self.use_rrag = use_rrag
        self.retrieval_aware = use_rrag
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        RRAGRunner.set_retrieval_token(RETRIEVAL_TOKEN)
        RRAGRunner.set_unk_token(UNK_TOKEN)
        RRAGRunner.set_unk_token_id(UNK_TOKEN_ID)
        RRAGRunner.set_instruction_type(instruction_type)

        self.num_k = num_k
        self.use_training = use_training
        self.freeze_llm = freeze_llm
        self.load_from_pretrained = load_from_pretrained
        self.pretrained_model_name = pretrained_model_name
        self.use_lora = use_lora
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size

        self.use_evaluation = use_evaluation
        self.max_new_tokens = max_new_tokens
        self.use_beam = use_beam
        self.beam_num = beam_num
        self.save_results = save_results

    @classmethod
    def set_unk_token(cls, token):
        cls.UNK_TOKEN = token

    @classmethod
    def set_retrieval_token(cls, token):
        cls.RETRIEVAL_TOKEN = token

    @classmethod
    def set_unk_token_id(cls, id):
        cls.UNK_TOKEN_ID = id

    @classmethod
    def set_instruction_type(cls, instruction_type):
        cls.instruction_type = instruction_type

    @staticmethod
    def format_instruction(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            if RRAGRunner.instruction_type == 'instruction':
                text = "### Instruction: \n{instruction}\n\n### Response:\n{output}".format(
                    instruction=example['instruction'][i],
                    output=example['output'][i]
                )
            elif RRAGRunner.instruction_type == 'chat':
                text = "[INST] {instruction} [/INST]{output}".format(
                    instruction=example['instruction'][i],
                    output=example['output'][i]
                )
            text = text.replace(RRAGRunner.RETRIEVAL_TOKEN, RRAGRunner.UNK_TOKEN)
            output_texts.append(text)
        return output_texts
    
    @staticmethod
    def format_instruction_for_response(prompt):
        if RRAGRunner.instruction_type == 'instruction':
            text = "### Instruction: \n{instruction}\n\n### Response:\n".format(
                instruction=prompt,
                )
        elif RRAGRunner.instruction_type == 'chat':
            text = "[INST] {instruction} [/INST]".format(
                instruction=prompt,
                )
        text = text.replace(RRAGRunner.RETRIEVAL_TOKEN, RRAGRunner.UNK_TOKEN)
        return text
    
    def load_tokenizer(self):
        if self.dataset_name == 'dureader': # Qwen
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=True, trust_remote_code=True)
            tokenizer.padding_side = "left"
            tokenizer.unk_token = '<|im_end|>'
            tokenizer.pad_token = tokenizer.eos_token
            RRAGRunner.set_unk_token_id(tokenizer.unk_token_id)
        else: # Llama
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=True)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

    def load_dataset(self):
        self.load_tokenizer()
        if self.dataset_name not in ['nq_10', 'nq_20', 'nq_30', 'hotpotqa', 'musique', '2wiki', 'dureader']:
            raise ValueError(self.dataset_name)
        if 'nq' in self.dataset_name:
            _load_dataset = load_nq_dataset
        elif self.dataset_name == 'hotpotqa' or self.dataset_name == '2wiki':
            _load_dataset = load_hotpotqa_dataset
        elif self.dataset_name == 'musique':
            _load_dataset = load_musique_dataset
        self.instruction_dataset_train, self.instruction_dataset_test = _load_dataset(self.input_path, self.max_prompt_length, self.tokenizer, retrieval_aware=self.retrieval_aware, RETRIEVAL_TOKEN=self.RETRIEVAL_TOKEN)
    
    def load_model(self):
        print('##############################  load_model  ##############################')
        if self.use_rrag:
            config = RRAGLlamaConfig(
                model_name_or_path=self.model_name,
                load_in_8bit=self.load_in_8bit,
                input_dim=self.input_dim,
                hidden_size=self.hidden_size,
                unk_token=self.UNK_TOKEN,
                unk_token_id=self.UNK_TOKEN_ID,
                freeze_llm=self.freeze_llm,
                num_k=self.num_k,
                )
            if self.load_from_pretrained:
                print(f'load_from_pretrained: {self.pretrained_model_name}')
                self.model = RRAGLlamaForCausalLM.from_pretrained(self.pretrained_model_name, config=config)
            else:
                self.model = RRAGLlamaForCausalLM(config)
        else:
            config = RAGLlamaConfig(
                model_name_or_path=self.model_name,
                load_in_8bit=self.load_in_8bit,
                freeze_llm=self.freeze_llm,
                )
            self.model = RAGLlamaForCausalLM(config)
        print(config)
        print(self.model)
    
    def get_peft_model(self):
        peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                inference_mode=False,
                task_type=TaskType.CAUSAL_LM, 
                target_modules=["q_proj", "v_proj"]
        )
        print(peft_config)
        self.model.llama_model = prepare_model_for_kbit_training(self.model.llama_model)
        self.model.llama_model = get_peft_model(self.model.llama_model, peft_config)
        self.model.llama_model.print_trainable_parameters()
        return peft_config
    
    def start_training(self):
        print('##############################  start_training  ##############################')
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            # disable_tqdm=True # disable tqdm since with packing values are in correct
        )
        dataset_train = Dataset.from_list(self.instruction_dataset_train[:])
        peft_config = self.get_peft_model() if self.use_lora else None
        max_seq_length = self.max_prompt_length
        trainer = RRAGTrainer(
            model=self.model,
            train_dataset=dataset_train,
            peft_config=peft_config,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            packing=False,
            formatting_func=RRAGRunner.format_instruction,
            args=args,
        )
        seed_it(42)
        trainer.train()
        # save model
        if self.save_model:
            print('output_dir', self.output_dir)
            self.model.save_model(self.output_dir)
        else:
            print('dont save_model')

    def get_response(self, sample, prompt_key='instruction'):
        prompt = RRAGRunner.format_instruction_for_response(sample[prompt_key])
        # print(prompt)
        input_tokens = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_prompt_length,
                    add_special_tokens=False,
                ).to('cuda')
        if self.use_rrag:
            embeds = torch.tensor(sample['embeds']).to(input_tokens.input_ids.device)
            label = torch.tensor(sample['label']).to(input_tokens.input_ids.device)
            if embeds.dim() == 1:
                embeds = embeds.unsqueeze(0)
            if label.dim() == 1:
                label = label.unsqueeze(0)
            inputs = {"input_ids": input_tokens['input_ids'], 'attention_mask': input_tokens['attention_mask'], 'embeds': embeds, 'label': label}
        else:
            inputs = {"input_ids": input_tokens['input_ids'], 'attention_mask': input_tokens['attention_mask']}

        outputs = self.model.generate(
            inputs=inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=self.beam_num if self.use_beam else 1,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
        )
        output_text = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
        output_text = [text.strip() for text in output_text]
        return output_text
    
    def eval(self):
        print('##############################  evaluation_from_list  ##############################')
        self.model.eval()
        self.model.llama_model.eval()
        res = []
        for data in tqdm(self.instruction_dataset_test[:], desc='get_response'):
            cur_res = self.get_response(data)[0]
            res.append(cur_res)
        if self.save_results:
            save_pkl_file = f'res_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if not os.path.exists('output'):
                os.makedirs('output')
            pkl_save_path = f'output/{save_pkl_file}.pkl'
            print('results_save_path', pkl_save_path)
            with open(pkl_save_path, 'wb') as f:
                pickle.dump(res, f)
                f.close()
        if 'nq' in self.dataset_name:
            get_ans = get_nq_ans
        elif self.dataset_name == 'hotpotqa' or self.dataset_name == '2wiki':
            get_ans = get_hotpotqa_ans
        elif self.dataset_name == 'musique':
            get_ans = get_musique_ans
        gt_ans = get_ans(self.instruction_dataset_test)
        m = evaluation_from_list(res[:], gt_ans[:len(res)], self.dataset_name)
    
    def run(self):
        self.load_dataset()
        self.load_model()
        print(self.use_training, self.use_evaluation)
        if self.use_training:
            self.start_training()
        if self.use_evaluation:
            self.eval()

def main(dataset_name, input_path, train_data_path, test_data_path, **args):
    if dataset_name in ['hotpotqa', 'musique', '2wiki']:
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
    print(args)
    runner = RRAGRunner(
        dataset_name=dataset_name,
        input_path=input_path, 
        **args
        )
    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with given parameters")
    
    parser.add_argument('--dataset_name', type=str, default='nq_10', choices=['nq_10', 'nq_20', 'nq_30', 'hotpotqa', 'musique', '2wiki', 'dureader'], help='Name of the dataset')
    parser.add_argument('--input_path', type=str, default=None, help='Path for nq or dureader datasets')
    parser.add_argument('--train_data_path', type=str, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--test_data_path', type=str, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--max_prompt_length', type=int, default=4096, help='Maximum prompt length')

    parser.add_argument('--model_name', type=str, required=True, help='Name of LLM')
    parser.add_argument('--load_in_8bit', action='store_true', help='Load in 8-bit precision')
    parser.add_argument('--save_model', action='store_true', help='If set, the trained model will be saved to outputdir')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save model')

    parser.add_argument('--use_rrag', action='store_true', help='Whether to use RRAG or not')
    parser.add_argument('--input_dim', type=int, default=3, help='Input features')
    parser.add_argument('--hidden_size', type=int, default=4096, help='Size of the LLM hidden layer')
    parser.add_argument('--RETRIEVAL_TOKEN', type=str, default='<R>', help='Token for retrieval')
    parser.add_argument('--UNK_TOKEN', type=str, default='<unk>', help='Token for unknown tokens')
    parser.add_argument('--UNK_TOKEN_ID', type=int, default=0, help='Unknown token ID')

    parser.add_argument('--num_k', type=int, default=10, help='Number of K in retrieval')
    parser.add_argument('--use_training', action='store_true', help='Use for training')
    parser.add_argument('--freeze_llm', action='store_true', help='Freeze LLM')
    parser.add_argument('--load_from_pretrained', action='store_true', help='Load from RRAG pretrained model')
    parser.add_argument('--pretrained_model_name', type=str, required=False, help='Name of RRAG pretrained model')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--num_train_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Batch size per device')

    parser.add_argument('--use_evaluation', action='store_true', help='Use for evaluation')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum new tokens')
    parser.add_argument('--use_beam', action='store_true', help='Use beam search')
    parser.add_argument('--beam_num', type=int, default=5, help='Number of beams in beam search')
    parser.add_argument('--save_results', action='store_true', help='Save results')
    parser.add_argument('--instruction_type', default='instruction', choices=['chat', 'instruction'], help='instruction_type, llama or mistral')

    args = parser.parse_args()
    main(**vars(args))
