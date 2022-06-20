# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# from fairseq.models.roberta import RobertaModel
# from fairseq import utils

from fairseq.models.roberta import RobertaModel
from fairseq import utils
import torch
from lama.modules.base_connector import *
from transformers import AutoTokenizer, BertTokenizer, BertEntForMaskedLM, BertConfig, RobertaTokenizer
# from lama.modules.model import LukeForMaskedLM
import json
import pickle
import numpy as np
import torch.nn.functional as F
import unicodedata
import os
import math
from .bert_connector import CustomBaseTokenizer
import urllib.parse



class BertConcat(Base_Connector):
    def __init__(self, args):
        super().__init__()
        bert_model_name = args.model_name_or_path
        # self.kind = args.kind
        do_lower_case = False
        if 'uncased' in bert_model_name:
            do_lower_case=True
        self.modL = args.modL
        self.max_sentence_length = args.max_sentence_length
        # self.add_prefix_space = args.add_prefix_space

        self.tokenizer = BertTokenizer.from_pretrained( bert_model_name,  do_lower_case=do_lower_case)
        # config = BertConfig.from_pretrained(  '/home/yedeming/bert_models/roberta-base' )

        self.map_indices = None
        self.vocab = list(self.tokenizer.ids_to_tokens.values())
        self._init_inverse_vocab()

        self.model =  BertEntForMaskedLM.from_pretrained( bert_model_name )
        self.model.bert.entity_embeddings.token_type_embeddings.weight.data.copy_(self.model.bert.embeddings.token_type_embeddings.weight.data)
        self.model.bert.entity_embeddings.LayerNorm.weight.data.copy_(self.model.bert.embeddings.LayerNorm.weight.data)
        self.model.bert.entity_embeddings.LayerNorm.bias.data.copy_(self.model.bert.embeddings.LayerNorm.bias.data)

        self.dim = dim = 768
        if self.modL>0:

            # try:
            #     if do_lower_case:
            #         embed_path = '/home/yedeming/PELT/wikiembed_bert/'
            #     else:
            #         embed_path = '/home/yedeming/PELT/wikiembed_bert_cased/'
            #     self.name2pageid =  json.load(open(embed_path+'name2id.json'))
            #     self.qid2pageid =  json.load(open(embed_path +'qid2pageid.json'))
            #     self.pageid2id = pickle.load(open(embed_path+'wiki_pageid2embedid.pkl', 'rb'))
            #     self.tot_entity_embed = np.load(embed_path + 'wiki_entity_embed_256.npy')
            #     L = np.linalg.norm(self.tot_entity_embed, axis=1)
            #     self.tot_entity_embed = self.tot_entity_embed / np.expand_dims(L,axis=-1) * self.modL
            #     self.dim = self.tot_entity_embed.shape[1]

            # except:
            assert( do_lower_case )

            embed_path = '/data3/private/yedeming/bert_entemb/'
            self.name2pageid =  json.load(open(embed_path+'name2id.json'))
            self.qid2pageid =  json.load(open(embed_path +'qid2pageid.json'))
            self.pageid2id = pickle.load(open(embed_path+'wiki_pageid2embedid.pkl', 'rb'))
            self.tot_entity_embed = np.load(embed_path + 'wiki_entity_embed_256.npy')
            L = np.linalg.norm(self.tot_entity_embed, axis=1)
            self.tot_entity_embed = self.tot_entity_embed / np.expand_dims(L,axis=-1) * self.modL
            self.dim = self.tot_entity_embed.shape[1]
        else:
            self.name2pageid = {}
            self.qid2pageid = {}

        # self.name2pageid = {}


        print ('modL', args.modL)



        # self.tot_entity_embed = np.memmap(embed_path+'tot_embed_mask.memmap', mode='r',  dtype=np.float16)

        # self.qid2ents = json.load(open('/home/yedeming/PELT/LAMA/qid2ents.json'))
        # num_example = self.tot_entity_embed.shape[0]//dim
        # self.tot_entity_embed = np.reshape(self.tot_entity_embed, (num_example, dim))

        print ('loaded')

    def _cuda(self):
        self.model.cuda()

    # def _is_subword(self, token):
    #     if isinstance(self.tokenizer, RobertaTokenizer):
    #         token = self.tokenizer.convert_tokens_to_string(token)
    #         if not token.startswith(" ") and not self._is_punctuation(token[0]):
    #             return True
    #     elif token.startswith("##"):
    #         return True

    #     return False

    # @staticmethod
    # def _is_punctuation(char):
    #     # obtained from:
    #     # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
    #     cp = ord(char)
    #     if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
    #         return True
    #     cat = unicodedata.category(char)
    #     if cat.startswith("P"):
    #         return True
    #     return False

    # @staticmethod
    # def _normalize_mention(text):
    #     return " ".join(text.split(" ")).strip()


    # def _detect_mentions(self, tokens, name):
    #     name = self._normalize_mention(name)

    #     for start, token  in enumerate(tokens):

    #         if self._is_subword(token):# and (start>1):
    #             continue
        
    #         for end in range(start, len(tokens)):
    #             if end < len(tokens) and self._is_subword(tokens[end]):# and (end>1):
    #                 continue
    #             mention_text = self.tokenizer.convert_tokens_to_string(tokens[start:end])
    #             mention_text = self._normalize_mention(mention_text)
    #             if len(mention_text) > len(name):
    #                 break
    #             if mention_text.lower()==name.lower():
    #                 return start, end


    #     return -1, -1


    # def _detech_mentions_squad(self, tokens, ent2id):
    #     mentions = []
    #     cur = 0
    #     for start, token  in enumerate(tokens):

    #         if start < cur:
    #             continue

    #         if self._is_subword(token) and (start>1):
    #             continue

    #         for end in range(min(start + 30, len(tokens)), start, -1):
    #             if end < len(tokens) and self._is_subword(tokens[end]) and (end>1):
    #                 continue

    #             mention_text = self.tokenizer.convert_tokens_to_string(tokens[start:end])
    #             mention_text = self._normalize_mention(mention_text)
    #             if mention_text in ent2id:

    #                 cur = end
    #                 pageid = ent2id[mention_text]

    #                 mentions.append((pageid, start, end))

    #                 break
                    


    #     return mentions


    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)

        return indexed_string

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True, sub_labels=None, sub_ids=None):
        # try_cuda = False   # for debug
        if not sentences_list:
            return None
        if torch.cuda.is_available():
            self.model.cuda()

            

        masked_indices_list = []
        max_len = 0
        output_tokens_list = []
        input_embeds_list = []
        attention_mask_list = []
        position_ids_list = []
        input_ids_list = []

        entity_embeddings_list = []
        entity_attention_mask_list = []
        entity_position_ids_list = []

        # pad_id = self.task.source_dictionary.pad()
        # pad_id = self.tokenizer.pad_id
        entity_K = 1
        if sub_ids is None:
            sub_ids = [-1]* len(sentences_list)
        for masked_inputs_list, sub_label, sub_id in zip(sentences_list, sub_labels, sub_ids):

            assert(len(masked_inputs_list)==1)
            for idx, masked_input in enumerate(masked_inputs_list):
                if sub_id in self.qid2pageid:
                    sub_pageid = self.qid2pageid[sub_id]
                else:
                    sub_label_align = urllib.parse.unquote(sub_label) # Added
                    if sub_label_align in self.name2pageid:
                        sub_pageid =  self.name2pageid[sub_label_align]
                    elif sub_label_align.lower() in self.name2pageid:
                        sub_pageid =  self.name2pageid[sub_label_align.lower()]
                    else:
                        sub_pageid = -1

                tokens = self.tokenizer.tokenize(masked_input)#, add_prefix_space=self.add_prefix_space)#, add_special_tokens=False)  not implement in 2.5.1
                tokens = [self.tokenizer.cls_token] + tokens #+ [self.tokenizer.sep_token]
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                mask_s = -1
                for k in range(len(tokens)):
                    if tokens[k]==MASK:
                        mask_s = k
                        break
                assert(mask_s!=-1)


                spliter_id = self.tokenizer.encode(' /', add_special_tokens=False)
                assert(len(spliter_id)==1)
                # spliter_token = self.tokenizer.tokenize(' /')
                l_id = self.tokenizer.encode(' (', add_special_tokens=False)
                assert(len(l_id)==1)
                r_id = self.tokenizer.encode(' )', add_special_tokens=False)
                assert(len(r_id)==1)


                output_tokens = []
                mentions = []
                if sub_label=='Squad':
                    assert (False)
                #     ents = self.qid2ents[sub_id]
                #     ent2id = {}
                #     for x in ents:
                #         if x[1]!='MASK':
                #             ent2id[self._normalize_mention(x[1])] = x[0]
                #     mentions = self._detech_mentions_squad(tokens, ent2id)

                if  sub_pageid>=0 and self.modL>0:
                    # sub_s, sub_e = self._detect_mentions(tokens, sub_label)
                    sub_s = 1
                    sub_e = sub_s + len(self.tokenizer.tokenize(sub_label))

                    # if( sub_s>=0 ):
                    # mentions = [(sub_pageid, sub_s, sub_e, sub_s)]
                    # input_ids = input_ids[:sub_s] + [self.tokenizer.mask_token_id] + spliter_id + input_ids[sub_s:]

                    # mask_s += 2

                    # # mentions = [(sub_pageid, sub_s, sub_e, sub_e+1)]
                    # # input_ids = input_ids[:sub_e] + spliter_id + [self.tokenizer.mask_token_id] + input_ids[sub_e:]

                    # mask_s += 2
                    
                    # mentions = [(sub_pageid, sub_s, sub_e, sub_s+1)]
                    # input_ids = input_ids[:sub_s] + l_id + [self.tokenizer.mask_token_id] + r_id + input_ids[sub_s:]
                    # mask_s += 3

                    # if self.kind==0:
                    mentions = [(sub_pageid, sub_s, sub_e, sub_e+1)]
                    input_ids = input_ids[:sub_e] + l_id + [self.tokenizer.mask_token_id] + r_id + input_ids[sub_e:]
                    mask_s += 3
                    # else:

                    #     mentions = [(sub_pageid, sub_s, sub_e, sub_s)]
                    #     input_ids = input_ids[:sub_s] +  [self.tokenizer.mask_token_id] + l_id + input_ids[sub_s:sub_e] + r_id + input_ids[sub_e:]
                    #     mask_s += 3

                    # else:
                    #     print (tokens, sub_label)
                    #     mentions = []

                input_ids = input_ids[:self.max_sentence_length-1] + [self.tokenizer.sep_token_id]
                entity_embeddings = []
                entity_position_ids = []
                # entity_attention_mask = []
                np.random.shuffle(mentions)

                for page_id, sub_s, sub_e, pos_ent in mentions:

                    if page_id in self.pageid2id:
                        embed_id = self.pageid2id[page_id]
                        # print (tokens[sub_s: sub_e])
                        entity_embedding = np.array(self.tot_entity_embed[embed_id], dtype=np.float32)
                        # entity_position_id = list(range(sub_s, sub_e))
                        # entity_position_id = entity_position_id[:30]
                        # entity_position_id = [sub_s]
                        # entity_position_id += [-1]*(30-len(entity_position_id))
                        # if self.modL>0:
                        #     L = np.linalg.norm(entity_embedding)
                        #     entity_embedding = entity_embedding / L * self.modL
                            
                        entity_embeddings.append(entity_embedding)
                        entity_position_ids.append(pos_ent)
                    if len(entity_embeddings)>=entity_K:
                        break

                # entity_attention_mask = [1] * len(entity_embeddings)

                L = len(input_ids)
                max_len = max(max_len, L)

                padding_length = self.max_sentence_length - L
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask = [1] * L + [0] * padding_length
                attention_mask += [1] * len(entity_position_ids) + [0] * (entity_K-len(entity_position_ids))
                for page_id, sub_s, sub_e, pos_ent in mentions:
                    attention_mask[pos_ent] = 0



                while len(entity_embeddings) < entity_K:
                    entity_embeddings.append(np.zeros((self.dim, ), dtype=np.float32))
                    entity_position_ids.append(0)
                    # entity_position_ids.append([-1] * 30)
                    # entity_attention_mask.append(0)


                output_tokens_list.append(np.array(input_ids, dtype=np.int64))

                input_ids_list.append(input_ids)

                attention_mask_list.append(attention_mask)
                masked_indices_list.append(mask_s)
                entity_embeddings_list.append(torch.tensor(entity_embeddings, dtype=torch.float32))
                entity_position_ids_list.append(entity_position_ids)



        input_ids_list = torch.tensor(input_ids_list, dtype=torch.int64)
        attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.int64)
        masked_indices_list = torch.tensor(masked_indices_list, dtype=torch.int64)

        entity_embeddings_list = torch.stack(entity_embeddings_list,  dim=0)
        entity_position_ids_list = torch.tensor(entity_position_ids_list,  dtype=torch.int64)

        with torch.no_grad():
            self.model.eval()
            if try_cuda:
                outputs = self.model(

                    input_ids=input_ids_list.cuda(),
                    attention_mask=attention_mask_list.cuda(),#.to(device=self.model.device),
                    entity_embeddings=entity_embeddings_list.cuda(),#(entity_embeddings_list/L_entity).cuda() * 3.64537835,
                    # entity_attention_mask=entity_attention_mask_list.cuda(),
                    entity_position_ids=entity_position_ids_list.cuda()

                )
            else:
                outputs = self.model(
                    input_ids=input_ids_list,
                    attention_mask=attention_mask_list,#.to(device=self.model.device),
                    entity_embeddings=entity_embeddings_list,
                    # entity_attention_mask=entity_attention_mask_list,
                    entity_position_ids=entity_position_ids_list
                )

            log_probs = outputs[0]



        return log_probs.cpu(), output_tokens_list, masked_indices_list.unsqueeze(1)

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        # TBA
        return None

