# -*- coding: utf-8 -*-


import argparse
import pickle
import subprocess
import time
import random
import pandas as pd
import numpy as np
from abc import ABC
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data.dataloader import Dataset, DataLoader
from transformers import set_seed, AutoConfig, XLMRobertaTokenizer
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel, RobertaForMaskedLM, RobertaEmbeddings, RobertaLMHead, RobertaEncoder
)
import logging
import os


os.environ['CURL_CA_BUNDLE'] = ''
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


DData = '../../data'
DOutput = '.'
model_name_or_path = os.path.join(DData, 'huggingface', 'xlm-roberta-base')


class BaseGameMaster:
    train_dataset = valid_dataset = test_dataset = None

    def __init__(self):
        self.gpu = -1
        self.manu_seed = 137
        self.learning_rate = 1e-3
        self.func = None
        self.exp_name = ''
        self.exp_id = ''
        self.exp_name_path = ''
        self.exp_id_path = ''

    def a(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])
        return self

    def add_dict(self, p_dict):
        for name in p_dict:
            setattr(self, name, p_dict[name])
        return self

    def get_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        if self.manu_seed > 0:
            random.seed(self.manu_seed)
            np.random.seed(self.manu_seed)
            torch.manual_seed(self.manu_seed)

        if int(self.gpu) < 0 or not torch.cuda.is_available():
            return torch.device("cpu")
        else:
            torch.cuda.manual_seed(self.manu_seed)
            return torch.device("cuda")

    def fdExpName(self):
        if self.exp_name_path == '':
            self.exp_name = 'debug' if self.exp_name == '' else self.exp_name
            self.exp_name_path = os.path.join(DOutput, self.exp_name)

        if not os.path.exists(self.exp_name_path):
            subprocess.Popen(f"mkdir {self.exp_name_path}", shell=True).wait()

        return self.exp_name_path

    def fdExpId(self):
        exp_name_path = self.fdExpName()

        if self.exp_id_path == '':
            if self.exp_id == '':
                chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
                while True:
                    exp_id = ''.join(random.choice(chars) for _ in range(10))
                    exp_id_path = os.path.join(exp_name_path, exp_id)
                    if not os.path.isdir(exp_id_path):
                        self.exp_id = exp_id
                        break
            else:
                exp_id_path = os.path.join(exp_name_path, self.exp_id)

            self.exp_id_path = exp_id_path

        if not os.path.isdir(self.exp_id_path):
            subprocess.Popen("mkdir %s" % self.exp_id_path, shell=True).wait()

        return self.exp_id_path

    def fNetState(self, _epoch=None):
        if _epoch is not None:
            return os.path.join(self.fdExpId(), f'net_state.ep{_epoch}.pt')
        return os.path.join(self.fdExpId(), f'net_state.pt')






def prefix_copy(_reference_state_dict, _src_prefix, _dst_prefix, _new_state_dict):
    _src_keys = []
    for _key in _reference_state_dict.keys():
        if _key.startswith(_src_prefix):
            _src_keys.append(_key)

    for _key in _src_keys:
        _new_state_dict[_dst_prefix + _key[len(_src_prefix):]] = _reference_state_dict[_key]


def get_path_marc_fewshot(*_names):
    dMarc_fewshot = os.path.join(DData, 'marc_fewshot')
    if len(_names) > 0:
        return os.path.join(dMarc_fewshot, *_names)
    return dMarc_fewshot


def loading_fewshot_marc(_lang_code, _mode=None, _shot=None, _seed=None):
    if _mode == "train":
        fRawCSV = get_path_marc_fewshot('raw', f'{_lang_code}.{_shot}_{_seed}.train.csv')
    elif _mode == "valid":
        fRawCSV = get_path_marc_fewshot('raw', f'{_lang_code}.{_shot}_{_seed}.dev.csv')
    elif _mode == "test":
        fRawCSV = get_path_marc_fewshot('raw', f'test.{_lang_code}.csv')
    else:
        raise NotImplementedError

    lines = pd.read_csv(fRawCSV, header=None).values.tolist()
    dict_texts, dict_labels = {}, {}
    for _i, line in enumerate(lines):
        guid = f"{_mode}_{_i}"
        dict_texts[guid] = line[1]
        dict_labels[guid] = line[2]

    return {'texts': dict_texts, 'labels': dict_labels}


def build_prompt_feature(
        text=None,
        label=None,
        _tokenizer=None,
        max_length=None,
        label_word_ids=None,  # {0: 11189, 1: 4676}
        _prompt_template=None,  # '*cls*_It_is_*mask*.*sentl_0**sep+*'
        prompt_token_ids=None,
):
    def enc(__text):
        return _tokenizer.encode(__text, add_special_tokens=False, max_length=max_length, truncation=True)

    if hasattr(_tokenizer, 'convert_token_to_id'):
        def word2id(__word):
            return _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(__word)[-1])
    elif hasattr(_tokenizer, '_convert_token_to_id'):
        word2id = _tokenizer._convert_token_to_id
    else:
        raise NotImplementedError

    if _prompt_template is None:
        input_ids = [_tokenizer.cls_token_id]
        attention_mask = [1]

        input_tokens = enc(text.lower()) + [_tokenizer.sep_token_id]

        input_ids += input_tokens
        attention_mask += [1 for _i in range(len(input_tokens))]

    else:
        special_token_mapping = {
            'cls': _tokenizer.cls_token_id,
            'sep': _tokenizer.sep_token_id,
            'sep+': _tokenizer.sep_token_id,
            'mask': _tokenizer.mask_token_id,
        }  # {'cls': 101, 'sep': 102, 'sep+': 102, 'mask': 103}

        is_encoded_prompt = False
        prompt_length = len(prompt_token_ids) if prompt_token_ids is not None else 0
        max_text_length = max_length - prompt_length - 4
        template_list = _prompt_template.split('*')

        input_ids = []
        attention_mask = []
        token_type_ids = []
        segment_id = 0

        for part_i, part in enumerate(template_list):
            new_tokens = []
            segment_plus_1_flag = False

            if part in special_token_mapping:
                new_tokens.append(special_token_mapping[part])
                if part == 'sep+':
                    segment_plus_1_flag = True

            elif part[:5] == 'sent_':
                new_tokens += enc(text)

            elif part[:6] == 'sentl_':
                new_tokens += enc(text.lower())

            elif part[:6] == 'prompt':
                # encode prompt
                is_encoded_prompt = True
                part = part[6:].replace('_', ' ')
                new_tokens += enc(part)
            else:
                part = part.replace('_', ' ')
                if len(part) == 1:
                    new_tokens.append(word2id(part))
                else:
                    new_tokens += enc(part)

            input_ids += new_tokens
            attention_mask += [1 for _ in range(len(new_tokens))]
            token_type_ids += [segment_id for _ in range(len(new_tokens))]
            if segment_plus_1_flag:
                segment_id += 1

            if not is_encoded_prompt and len(input_ids) > max_text_length:
                input_ids = input_ids[:max_text_length]
                attention_mask = attention_mask[:max_text_length]
                token_type_ids = token_type_ids[:max_text_length]

    while len(input_ids) < max_length:
        input_ids.append(_tokenizer.pad_token_id)  # [0, 1650, 25, 7, 250001, 6, 5, 2, 1, 1, 1, ]
        attention_mask.append(0)  # [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0...
        # token_type_ids.append(0)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...

    # truncation
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        # token_type_ids = token_type_ids[:max_length]

    if label is None:
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    prompt_pos = -1
    if prompt_token_ids is not None:
        prompt_pos = [input_ids.index(p_id) for p_id in prompt_token_ids]

    # mask position
    mask_pos = [input_ids.index(_tokenizer.mask_token_id)]
    assert mask_pos[0] < max_length

    mlm_labels = [-100 for _ in range(len(input_ids))]
    mlm_labels[mask_pos[0]] = label_word_ids[label]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'mask_pos': mask_pos,
        'label': mlm_labels,
        'label_word_ids': list(label_word_ids.values()),
        'prompt_pos': prompt_pos
    }


def build_prompt_feature_collate_fn(batch_data):
    input_ids = torch.tensor([item['input_ids'] for item in batch_data]).long()  # (bs, len)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch_data]).long()  # (bs, len)
    mask_pos = torch.tensor([item['mask_pos'] for item in batch_data]).long().squeeze()  # (bs, )
    labels = torch.tensor([item['label'] for item in batch_data]).long()  # (bs, len)
    label_word_ids = torch.tensor(batch_data[0]['label_word_ids']).long()  # (num_labels, )
    prompt_pos = torch.tensor(batch_data[0]['prompt_pos']).long()  # (bs, )

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'mask_pos': mask_pos,
        'labels': labels,
        'label_word_ids': label_word_ids,
        'prompt_pos': prompt_pos
    }


class RobertaPromptDatasetForFewshotMarc(Dataset):
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    def __init__(
            self,
            gm,
            _lang_code,  # {str} 'en'
            _mode=None,  # {str} 'test'
            _num_data=None,
    ):
        if _mode == "train":
            fPromptDataset = get_path_marc_fewshot(f'{gm.model_name}.{gm.prompt_code}'
                                                   f'.{_lang_code}.{gm.shot}_{gm.seed}.train.pkl')
        elif _mode == "valid":
            fPromptDataset = get_path_marc_fewshot(f'{gm.model_name}.{gm.prompt_code}'
                                                   f'.{_lang_code}.{gm.shot}_{gm.seed}.valid.pkl')
        elif _mode == "test":
            fPromptDataset = get_path_marc_fewshot(f'{gm.model_name}.{gm.prompt_code}.{_lang_code}.test.pkl')
        else:
            raise NotImplementedError

        if os.path.exists(fPromptDataset) and not gm.is_rebuild:
            self.features = pickle.load(open(fPromptDataset, 'rb'))
        else:
            self.features = []
            dict_rawtexts, dict_labels = loading_fewshot_marc(_lang_code, _mode, gm.shot, gm.seed).values()

            for guid, _raw_text in dict_rawtexts.items():
                self.features.append(build_prompt_feature(
                    _raw_text, dict_labels[guid],
                    gm.tokenizer, gm.max_len,
                    gm.labels_word_ids,
                    gm.prompt_template, gm.prompt_token_ids
                ))

            pickle.dump(self.features, open(fPromptDataset, 'wb'))

        if _num_data is not None and _num_data != -1:
            random_indices = random.sample(range(len(self.features)), _num_data)
            self.features = [self.features[_i] for _i in random_indices]


class LabelAttnLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size  # 768
        self.label_attn_type = config.label_attn_type
        if self.label_attn_type == 'labat3':
            self.fc_Q = nn.Linear(hidden_size, hidden_size)
            self.fc_K = nn.Linear(hidden_size, hidden_size)
            self.fc_V = nn.Linear(hidden_size, hidden_size)
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(0.5)
            self.layer_norm = nn.LayerNorm(hidden_size)

        else:
            raise NotImplementedError

    def forward(self, word_encoded, mask_pos=None):
        loss_label_attn = -1
        Q = self.fc_Q(word_encoded)  # (b, s, e)
        K = self.fc_K(word_encoded)  # (b, s, e)
        V = self.fc_V(word_encoded)  # (b, s, e)
        Alpha = torch.matmul(Q, K.permute(0, 2, 1))  # (b, s, s)
        Alpha = F.softmax(Alpha, dim=-1)  # (b, s, s)
        contexts = torch.matmul(Alpha, V)  # (b, s, e)
        out = self.fc1(contexts)  # (b, s, e)
        out = self.dropout(out)
        word_encoded = word_encoded + out
        word_encoded = self.layer_norm(word_encoded)  # (b, s, e)
        return {'loss_label_attn': loss_label_attn, 'word_encoded': word_encoded}


class PCLEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = RobertaEmbeddings(config)
        self.prompt_length = config.prompt_length
        hidden_size = config.hidden_size  # 768
        self.prompt_ids = torch.tensor(list(range(self.prompt_length))).long()
        self.prompt_embeddings = torch.nn.Embedding(config.prompt_length, hidden_size)  # Embedding(2, 768)

        if config.prompt_encoder_type == "mlp":
            self.prompt_encoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size)
            )
        else:
            raise NotImplementedError

    def forward(
            self,
            input_ids,  # torch.Size([8, 128])
            prompt_pos  # torch.Size([2])
    ):
        raw_embeds = self.embeddings.word_embeddings(input_ids)  # torch.Size([8, 128, 768])
        prompt_ids = self.prompt_ids.to(input_ids.device)
        embeds = self.prompt_embeddings(prompt_ids)  # torch.Size([2, 768])
        embeds = embeds.unsqueeze(0)  # [batch_size, prompt_length, hidden_size]
        prompt_embeds = self.prompt_encoder(embeds)  # torch.Size([1, 2, 768])

        for batch_i, item in enumerate(raw_embeds):
            for p_i, p_pos in enumerate(prompt_pos):
                raw_embeds[batch_i][p_pos] = prompt_embeds[0, p_i]

        embedding_output = self.embeddings(inputs_embeds=raw_embeds)  # (b, s, e)=torch.Size([8, 128, 768])
        return embedding_output


class PCLModel(RobertaPreTrainedModel, ABC):
    @staticmethod
    def loading_model(gm):
        _tokenizer = XLMRobertaTokenizer.from_pretrained(model_name_or_path)
        auto_config = AutoConfig.from_pretrained(model_name_or_path)
        auto_config.num_labels = gm.num_labels

        new_states = {}
        roberta = RobertaForMaskedLM.from_pretrained(model_name_or_path, config=auto_config)
        roberta_states = roberta.state_dict()
        prefix_copy(roberta_states, 'roberta.embeddings.', 'pcl_embedding.embeddings.', new_states)
        prefix_copy(roberta_states, 'roberta.encoder.layer.', 'encoder.layer.', new_states)
        prefix_copy(roberta_states, 'lm_head.', 'lm_head.', new_states)
        torch.manual_seed(137)
        prompt_embeddings = torch.nn.Embedding(gm.prompt_length, auto_config.hidden_size)
        new_states['pcl_embedding.prompt_embeddings.weight'] = prompt_embeddings.weight

        auto_config.device = gm.get_device()
        auto_config.batch_size = gm.batch_size
        auto_config.max_len = gm.max_len
        auto_config.prompt_length = gm.prompt_length
        auto_config.gradient_checkpointing = True
        auto_config.prompt_encoder_type = gm.prompt_encoder_type
        auto_config.alpha = gm.alpha
        auto_config.label_attn_type = gm.label_attn_type
        auto_config.train_style = gm.train_style
        auto_config.unuse_rate = gm.unuse_rate
        _network_model = PCLModel.from_pretrained(None, state_dict=new_states, config=auto_config)
        return _tokenizer, _network_model

    def get_input_embeddings(self):
        return self.pcl_embedding.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.pcl_embedding.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def __init__(self, config):
        super().__init__(config)

        self.alpha = config.alpha
        self.label_attn = None
        if 0 < config.alpha <= 1:
            self.label_attn = LabelAttnLayer(config)

        if self.label_attn is None:
            logger.debug("[PCLModel. No label_attn]")
        else:
            logger.debug(f"[PCLModel. {self.alpha} label_attn]")

        self.pcl_embedding = PCLEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()
        self.num_labels = config.num_labels
        self.loss_nll = nn.NLLLoss(ignore_index=-100)

        self.unuse_rate = config.unuse_rate
        hidden_size = config.hidden_size
        self.fea2label_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * config.max_len, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.batch_use_layer = torch.nn.Linear(hidden_size * config.max_len, 1)
        self.loss_mse = torch.nn.MSELoss()

        if config.train_style == 'selflr':
            self.forward = self.forward_selflr

    def forward_selflr(
            self,
            input_ids=None,  # torch.Size([8, 128])=tensor([0, 250002, 250003, 250001, 6, 5, 581, 10576, 83, ...])
            attention_mask=None,  # torch.Size([8, 128])=tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1...], device='cuda:0')
            mask_pos=None,  # tensor([3, 3, 3, 3, 3, 3, 3, 3], device='cuda:0')
            labels=None,  # torch.Size([8, 128])=tensor([ -100, -100, -100, 68403, -100, -100, -100, ...
            label_word_ids=None,  # tensor([94176,  6494, 68403,  4127,  6782], device='cuda:0')
            prompt_pos=None,  # tensor([1, 2], device='cuda:0')
            is_for_soft_label=None,
            similarity_order=None,  # tensor([10,  7,  0,  3,  6, 14,  1, 15,  8, 12,  9, 11,  4, 13,  5,  2])
            selflearning='inference',
            **kwargs
    ):
        batch_size, sequence_length, device = input_ids.shape[0], input_ids.shape[-1], input_ids.device

        embedding_output = self.pcl_embedding(input_ids, prompt_pos)  # (b, s, e)

        extended_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_ids.shape)  # (8,1,1,128)
        word_encoded = self.encoder(embedding_output, attention_mask=extended_mask, return_dict=False)
        word_encoded = word_encoded[0]  # (b, s, e) = torch.Size([8, 128, 768])

        if selflearning == 'inference':
            num_use = batch_size
            encoded_use = word_encoded
            labels_use = labels
            loss_fea2label = -1

        else:
            label_vectors = [word_encoded[_i, mask_pos[_i], :] for _i in range(batch_size)]  # list[torch.Size([768])]
            label_vectors = torch.stack(label_vectors, dim=0)  # torch.Size([b, e])
            pseudo_label = self.fea2label_layer(word_encoded.view(batch_size, -1))  # (b, e)

            if similarity_order is None:
                similarities = torch.cosine_similarity(label_vectors, pseudo_label)
                similarity_order = torch.argsort(similarities, dim=-1, descending=True)
                return {'similarities': similarities, 'similarity_order': similarity_order}

            loss_fea2label = self.loss_mse(pseudo_label, label_vectors)
            num_use = len(similarity_order) - self.unuse_rate
            encoded_use = torch.index_select(word_encoded, 0, similarity_order[:num_use])  # (13, s, e)
            # encoded_unuse = torch.index_select(word_encoded, 0, similarity_order[-num_unuse:])  # (3, s, e)
            labels_use = torch.index_select(labels, 0, similarity_order[:num_use])

        loss_label_attn = -1
        if self.label_attn is not None:
            label_attns = self.label_attn(encoded_use, mask_pos)
            loss_label_attn = label_attns['loss_label_attn']
            encoded_use = label_attns['word_encoded']

        prediction_scores = self.lm_head(encoded_use)  # (bs, len, vocab=250002)

        vocab_size = prediction_scores.shape[-1]
        prediction_logits = torch.zeros(num_use, sequence_length, vocab_size).to(device)  # (bs, len, vocab)
        for j in range(num_use):
            current_scores = prediction_scores[j]  # (len, vocab)
            masked_position = None

            logits = []
            for i_label in range(self.num_labels):
                masked_position = mask_pos[j]  # 3
                logits.append(current_scores[masked_position, label_word_ids[i_label]].unsqueeze(-1))

            logits = torch.cat(logits, -1)  # {list: num_labels} -> {Tensor: num_labels}
            logits_softmax = F.log_softmax(logits, -1)  # {Tensor: num_labels}

            mlm_logits = torch.zeros(vocab_size).to(device)  # {Tensor: vocab}
            for i_label in range(self.num_labels):
                mlm_logits[label_word_ids[i_label]] = logits_softmax[i_label]

            prediction_logits[j, masked_position] = mlm_logits

        loss = self.loss_nll(prediction_logits.view(-1, vocab_size), labels_use.view(-1))

        label_mask = torch.ones(1, vocab_size, dtype=torch.bool).to(device)  # torch.Size([1, 250002])
        for word_id in label_word_ids:
            label_mask[0][word_id] = False

        prediction_logits_for_matrics = torch.clone(prediction_logits)  # (bs, len, vocab)
        for j in range(num_use):
            prediction_logits_for_matrics[j] = prediction_logits[j].masked_fill(label_mask, -float("inf"))

        prediction = prediction_logits_for_matrics.argmax(-1)  # (bs, len)=(8, 128)

        loss_all = loss
        if loss_fea2label != -1 and loss_label_attn != -1:
            loss_all = loss * (1 - self.alpha) + loss_label_attn * self.alpha + loss_fea2label
        elif loss_fea2label != -1:
            # loss_all = loss + loss_fea2label
            loss_all = loss * (1 - self.alpha) + loss_fea2label * self.alpha
        elif loss_label_attn != -1:
            loss_all = loss * (1 - self.alpha) + loss_label_attn * self.alpha

        return {'loss': loss_all, 'prediction': prediction}


class PCLPlayer:
    def __init__(self, config, _model):
        config.exp_name_path = ''
        config.exp_id_path = ''
        self.gm = config
        self.netwk = _model
        self.acc = config.acc if hasattr(config, 'acc') else 'acc'  # acc f1

        self.train_loader = self.valid_loader = self.test_loader = None
        if config.train_dataset is not None:
            self.train_loader = DataLoader(
                dataset=config.train_dataset, collate_fn=config.collate_fn,
                batch_size=config.batch_size, num_workers=8, prefetch_factor=4, pin_memory=True, shuffle=True)

        if config.valid_dataset is not None:
            self.valid_loader = DataLoader(
                dataset=config.valid_dataset, collate_fn=config.collate_fn,
                batch_size=config.batch_size, num_workers=8, prefetch_factor=4, pin_memory=True, shuffle=True)

        if config.test_dataset is not None:
            self.test_loader = DataLoader(
                dataset=config.test_dataset, collate_fn=config.collate_fn,
                batch_size=config.batch_size, num_workers=4, prefetch_factor=4)  # , pin_memory=True

        if config.train_style == 'selflr':
            self.training_process = self.training_process_self_learning
            logger.debug('[PCLPlayer.self_learning]')
        else:
            raise NotImplementedError

    def testing_process(self, **kwargs):
        if not os.path.exists(self.gm.fNetState()):
            return None
        torch.cuda.empty_cache()

        test_loader = kwargs.pop("data_loader", self.test_loader)

        if test_loader is not None:
            self.netwk.load_state_dict(torch.load(self.gm.fNetState()))
            test_outputs = self.evaluate_process(
                device=self.gm.get_device(), model=self.netwk, data_loader=test_loader, **kwargs)
            return test_outputs
        else:
            return None

    @staticmethod
    def evaluate_process(*_args, **kwargs):
        device = kwargs.pop("device", None)
        model = kwargs.pop("model", None)
        data_loader = kwargs.pop("data_loader", None)
        desc = kwargs.pop("desc", None)

        if desc is not None and len(data_loader.dataset) > 32:
            _data_iter = tqdm(data_loader, desc=desc, ncols=80)
        else:
            _data_iter = data_loader

        all_loss, all_predicts, all_labels = [], [], []

        model.to(device)
        model.eval()
        with torch.no_grad():
            for step, inputs in enumerate(_data_iter):
                all_labels.append(inputs['labels'])
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                all_loss.append(float(outputs['loss']))
                all_predicts.append(outputs['prediction'])

        average_loss = np.mean(all_loss)
        all_predicts = torch.cat(all_predicts).cpu().numpy()
        all_labels = np.concatenate(all_labels)
        label_mask = (all_labels != -100)

        acc = (all_labels[label_mask] == all_predicts[label_mask]).mean()
        f1 = f1_score(all_labels[label_mask], all_predicts[label_mask], average='macro')
        return {
            'loss': average_loss,
            'acc': acc,
            'f1': f1
        }

    def training_process_self_learning(self):
        from transformers import get_linear_schedule_with_warmup

        gm = self.gm
        device = gm.get_device()
        num_epochs = gm.num_epochs
        num_steps = len(self.train_loader)
        total_steps = num_epochs * num_steps
        netwk = self.netwk.to(device)
        acc = self.acc
        stop_after_best_epoch = gm.stop_after_best_epoch if hasattr(gm, 'stop_after_best_epoch') else 5

        optimizer = torch.optim.AdamW(netwk.parameters(), lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=3, num_training_steps=total_steps)

        start_time, _train_outputs = time.time(), {}
        logger.info("[{}], {}".format(torch.device(torch.cuda.current_device()), gm.exp_id))
        pbar = tqdm(total=total_steps, mininterval=3, ncols=110,
                    bar_format='Train-'+gm.src_lang+': [{elapsed}<{remaining}]{postfix} --')
        epoch_loss, valid_loss = [], []
        valid_ls, valid_acc, best_epoch, best_loss, best_acc = 0.0, 0.0, 0, float('inf'), 0.0

        try:
            for epoch in range(1, num_epochs + 1):
                batch_loss = []
                netwk.train()
                for step, inputs in enumerate(self.train_loader):
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        with amp.autocast():
                            outputs = netwk(selflearning='train', **inputs)

                    similarity_order = outputs['similarity_order']
                    outputs = netwk(selflearning='train', **inputs, similarity_order=similarity_order)
                    loss = outputs['loss']

                    netwk.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    batch_loss.append(float(loss))
                    pbar.set_postfix_str(
                        f'Epoch={epoch}/{num_epochs}_{step + 1}/{num_steps}, '
                        f'lr={scheduler.get_last_lr()[0]:.8f}, loss={float(loss):.3f}, '
                        f'vls={valid_ls:.3f}, v{acc}={valid_acc:.3f}'
                    )
                    pbar.update()

                epoch_loss.append(np.mean(batch_loss))
                evaluate_outputs = self.evaluate_process(device=device, model=netwk, data_loader=self.valid_loader)
                valid_ls, valid_acc = evaluate_outputs['loss'], evaluate_outputs[acc]
                valid_loss.append(valid_ls)
                if valid_ls < best_loss:
                    best_epoch, best_loss, best_acc = epoch, valid_ls, valid_acc
                    torch.save(netwk.state_dict(), gm.fNetState())
                    with open(gm.fNetState() + '.txt', 'a') as f:
                        f.write(
                            f'Epoch={epoch}/{num_epochs}, lr={scheduler.get_last_lr()[0]:.8f}, '
                            f'train_loss={epoch_loss[-1]:.5f}, '
                            f'valid_ls={valid_ls:.5f}, v{acc}={valid_acc:.5f}\n'
                        )

                if epoch - best_epoch > stop_after_best_epoch:
                    pbar.close()
                    logger.debug(f"Early Stop. Best epoch={best_epoch}")
                    break

                if epoch >= 8:
                    aaa = np.mean(epoch_loss[-8:-4]) - np.mean(epoch_loss[-4:])
                elif epoch >= 6:
                    aaa = np.mean(epoch_loss[-6:-3]) - np.mean(epoch_loss[-3:])
                elif epoch >= 4:
                    aaa = np.mean(epoch_loss[-4:-2]) - np.mean(epoch_loss[-2:])
                else:
                    aaa = 1

                if 0 < aaa < 0.001:
                    pbar.close()
                    logger.debug(f"Early Stop. No optimization for a long time. last improvement={aaa:.5f}")
                    break

        except KeyboardInterrupt:
            pbar.close()
            if len(epoch_loss) > len(valid_loss):
                epoch_loss = epoch_loss[:-1]
            logger.debug("KeyboardInterrupt.")
        finally:
            pbar.close()

        _train_outputs.update({
            'time_cost': time.time() - start_time,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'best_acc': best_acc,
            'exp_id': gm.exp_id,
            'loss_dict': {
                'epoch_loss': np.asarray(epoch_loss),
                'valid_loss': np.asarray(valid_loss)
            }
        })
        return _train_outputs


"""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
"""


def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='PCL')
    parser.add_argument('--train_style', default='selflr')
    parser.add_argument('--prompt_encoder_type', default='mlp')
    parser.add_argument('--data_scale', default='fewshot')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--label_attn_type', default='labat3')
    parser.add_argument('--unuse_rate', type=int, default=1)

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--num_test', type=int, default=128, help="-1 means ALL")

    # --prompt_style=p1 --prompt_length=8 --prompt_code=p1l8_eng1
    parser.add_argument('--prompt_style', default='p1')
    parser.add_argument('--prompt_length', type=int, default=8)
    parser.add_argument('--prompt_code', default='p1l8_eng1')
    parser.add_argument('--src_lang', default='en')
    parser.add_argument('--tgt_lang', default='de')
    parser.add_argument('--shot', type=int, default=4, help="4 8 16 32 64 128 256")
    parser.add_argument('--seed', type=int, default=87, help="13 21 42 87 100")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--acc', default='acc')
    parser.add_argument('--optimizer_func', default='AdamW')
    parser.add_argument('--scheduler_step_size', type=int, default=1)
    parser.add_argument('--stop_after_best_epoch', type=int, default=5)
    parser.add_argument('--is_rebuild', action="store_true", default=False)
    parser.add_argument('--no_train', action="store_true", default=False)
    parser.add_argument('--no_src_test', action="store_true", default=False)
    args = parser.parse_args()
    bgm = BaseGameMaster().add_dict(args.__dict__)
    bgm.add_dict(kwargs)
    bgm.do_train = not bgm.no_train
    bgm.do_src_test = not bgm.no_src_test
    set_seed(13)
    logger.info('')
    logger.info(f'======== Initialized logger - {bgm.model_name} ========')

    en_labels_mapping = eval("{1:'terrible', 2:'bad', 3:'okay', 4:'good', 5:'great'}")  # {dict: 5}
    bgm.num_labels = 5

    tokenizer, network_model = PCLModel.loading_model(bgm)
    bgm.collate_fn = build_prompt_feature_collate_fn
    bgm.tokenizer = tokenizer
    bgm.prompt_token_ids = None

    en_labels_set = bgm.label_set = list(en_labels_mapping.keys())
    en_labels_words = list(en_labels_mapping.values())
    en_labels_word_ids_list = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)[-1])
                               for word in en_labels_words]
    en_labels_word_ids = {k: v for k, v in zip(en_labels_set, en_labels_word_ids_list)}

    artificial_tokens = []
    for i in range(bgm.prompt_length):
        artificial_tokens.append(f'[p{i}]')

    num_added_tokens = tokenizer.add_tokens(artificial_tokens)
    network_model.resize_token_embeddings(len(tokenizer))
    prompt_token_ids = tokenizer.convert_tokens_to_ids(artificial_tokens)
    bgm.prompt_tokens = artificial_tokens
    bgm.prompt_token_ids = prompt_token_ids

    if bgm.prompt_style == 'p1':
        bgm.prompt_template = f"*cls*prompt{' '.join(artificial_tokens)}*mask*._*sent_0**sep+*"
    else:
        raise NotImplementedError

    if bgm.prompt_code in ('p1l8_eng1', ):
        bgm.labels_mapping = en_labels_mapping
        bgm.labels_word_ids = en_labels_word_ids
    else:
        raise NotImplementedError

    if bgm.data_scale == 'fewshot':
        bgm.train_dataset = RobertaPromptDatasetForFewshotMarc(bgm, bgm.src_lang, 'train')

    bgm.valid_dataset = RobertaPromptDatasetForFewshotMarc(bgm, bgm.src_lang, 'valid')
    bgm.test_dataset = RobertaPromptDatasetForFewshotMarc(bgm, bgm.src_lang, 'test', bgm.num_test)

    bgm.exp_name = f'{bgm.model_name}_{bgm.src_lang}_{bgm.prompt_code}'
    bgm.exp_id = '{}_{}_{}_{}_ep{}bs{}len{}_{}_{}{}_u{}_{}'.format(
        bgm.src_lang, bgm.tgt_lang, bgm.shot, bgm.seed,
        bgm.num_epochs, bgm.batch_size, bgm.max_len,
        bgm.prompt_encoder_type,
        bgm.alpha, bgm.label_attn_type, bgm.unuse_rate,
        bgm.prompt_code
    )

    player = PCLPlayer(bgm, network_model)
    train_outputs = src_outputs = tgt_outputs = None

    if bgm.do_train:
        train_outputs = player.training_process()

    if bgm.do_src_test:
        src_outputs = player.testing_process(desc=f'Test-{bgm.src_lang}')

    if hasattr(bgm, 'tgt_lang') and bgm.tgt_lang is not None:
        torch.cuda.empty_cache()

        tgt_test_loader = DataLoader(
            dataset=RobertaPromptDatasetForFewshotMarc(bgm, bgm.tgt_lang, 'test', bgm.num_test),
            collate_fn=bgm.collate_fn,
            batch_size=bgm.batch_size, num_workers=2, prefetch_factor=4)

        tgt_outputs = player.testing_process(data_loader=tgt_test_loader, desc=f'Test-{bgm.tgt_lang}')

    logger.info('acc, {}={:.3f}, {}={:.3f}, f1, {}={:.3f}, {}={:.3f}, bv={:2d}_{:.3f}_{:.3f}, {:.0f}s, {}'.format(
        bgm.src_lang, src_outputs['acc'],
        bgm.tgt_lang, tgt_outputs['acc'],
        bgm.src_lang, src_outputs['f1'],
        bgm.tgt_lang, tgt_outputs['f1'],
        train_outputs['best_epoch'], train_outputs['best_loss'], train_outputs['best_acc'],
        train_outputs['time_cost'], train_outputs['exp_id']
    ))


if __name__ == '__main__':
    main()

"""
python run_pcl.py --gpu=0 --num_epochs=2 --num_test=128 --src_lang=en --tgt_lang=de --shot=8 --seed=87
python run_pcl.py --gpu=1 --num_epochs=25 --num_test=-1 --src_lang=en --tgt_lang=de --shot=8 --seed=87
"""

