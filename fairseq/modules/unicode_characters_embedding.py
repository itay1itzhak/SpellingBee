# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional

import torch
import torch.onnx.operators
from fairseq import utils
from torch import Tensor, nn
import torch.nn.functional as F


class UnicodeCharactersEmbedding(nn.Module):
    # class UnicodeCharactersEmbedding(nn.Embedding):
    """This module produces unicode characters embeddings of any length.

    Padding symbols are ignored.
    """

    def get_chars_in_vocab(self, vocab):
        chars_in_vocab = set()
        for token in vocab:  # .symbols:
            for c in token:
                if c not in chars_in_vocab:
                    chars_in_vocab.add(str(c))
        return sorted(list(chars_in_vocab))

    def char_emb(self, char_index):
        if self.embd_scale:
            return self.char_rand_embedding(torch.tensor(char_index).cuda()) / math.sqrt(self.embedding_dim)
        else:
            return self.char_rand_embedding(torch.tensor(char_index).cuda())

    def expi(self, token, double_char, without_shtrudel, word_marking, lang_marking):
        if word_marking:
            raise Exception("Not implemnted in this version, revert to told code")
        if lang_marking:
            raise Exception("Not implemnted in this version, revert to told code")

        if without_shtrudel:
            if token is not '@' and token is not '@@':
                token = token.replace('@', '')
        if self.one_shtrudel:
            token = token.replace('@@', '@')

        if double_char:
            dup_token = str()
            for c in token:
                dup_token += str(c) + str(c)
            token = dup_token

        return token

    def is_j_out_of_bounds(self, j):
        if self.rand_embedding:
            return self.embedding_dim // self.char_rand_emb_size < j + 1
        if j >= torch.div(self.embedding_dim, self.bits_need_per_char,
                          rounding_mode='floor'):  # no more room for more chars in this emb
            return True
        if (self.geresh_padding or self.super_geresh_padding) and j >= torch.div(self.embedding_dim,
                                                                                 (3 * self.bits_need_per_char),
                                                                                 rounding_mode='floor'):
            return True
        if (self.word_marking or self.lang_marking) and j + 2 >= torch.div(self.embedding_dim, self.bits_need_per_char,
                                                                           rounding_mode='floor'):  # 2 instead of 1 in case both are True
            return True
        if (self.geresh_padding or self.super_geresh_padding) and (
                self.word_marking or self.lang_marking) and j + 2 >= torch.div(
            self.embedding_dim, (3 * self.bits_need_per_char), rounding_mode='floor'):
            return True
        return False

    def get_index(self, j, k, bits_need_per_char):
        char_space_to_skip = j
        if self.geresh_padding or self.super_geresh_padding:
            char_space_to_skip = j * 3
        if self.word_marking or self.lang_marking:
            char_space_to_skip = j + 1
            if self.geresh_padding or self.super_geresh_padding:
                char_space_to_skip = j * 3 + 1
            if self.word_marking and self.lang_marking:
                char_space_to_skip = j + 2
                if self.geresh_padding or self.super_geresh_padding:
                    char_space_to_skip = j * 3 + 2

        index = -(k + bits_need_per_char * char_space_to_skip) - 1
        return index

    def set_confi(self):
        print("Vocab emb addition")
        self.double_char = False
        self.word_marking = False
        self.geresh_padding = False
        self.super_geresh_padding = False
        self.lang_marking = False
        self.lang_marking_0_5 = False
        self.without_shtrudel = False
        self.one_shtrudel = True
        self.rand_embedding = True
        self.embd_scale = False
        self.learn_char_rand_embedding = False

        if self.word_marking:
            print("full word marking @")
        if self.lang_marking:
            import re
            print("self.lang_marking")
        if self.without_shtrudel:
            print("self.without_shtrudel @")
        if self.double_char:
            print("with self.double_char")
        if self.one_shtrudel:
            print("self.one_shtrudel")
        if self.geresh_padding:
            print("self.geresh_padding")
        if self.super_geresh_padding:
            print("self.super_geresh_padding")
        if self.rand_embedding:
            print("self.rand_embedding")
        if self.learn_char_rand_embedding:
            print("self.learn_char_rand_embedding")
        if self.embd_scale:
            print("self.embd_scale")

    def __init__(self, embedding_dim, padding_idx, init_size=1024, vocab=None, embed_chars=None):
        super().__init__()
        # super().__init__(num_embeddings=embed_chars.weight.shape[0], embedding_dim=embed_chars.weight.shape[1])

        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.set_confi()

        if embed_chars is not None:
            self.char_rand_embedding = embed_chars.cuda()
            if not self.learn_char_rand_embedding:
                self.char_rand_embedding.weight.requires_grad = False
                print("char_rand_embedding is Not Learned")

            self.char_rand_emb_size = self.char_rand_embedding.weight.shape[1]

            if self.char_rand_embedding is not None:
                self.chars_in_vocab = self.get_chars_in_vocab(vocab.symbols)
                torch.nn.init.constant_(self.char_rand_embedding.weight[0], 0)  # char padding
                self.tokens_chars_idx = torch.zeros(len(vocab.symbols), embedding_dim // self.char_rand_emb_size,
                                                    dtype=torch.int)
                for i, token in enumerate(vocab.symbols):
                    if i < 5:  # for unique tokens <s>, <pad>, </s>, <unk>
                        continue
                    token = self.expi(token, self.double_char, self.without_shtrudel, self.word_marking,
                                      self.lang_marking)

                    for j, c in enumerate(token):
                        if self.is_j_out_of_bounds(j):
                            break
                        self.tokens_chars_idx[i, -j - 1] = self.chars_in_vocab.index(c) + 1  # (index 0 is char padding)

        self.padding_idx = padding_idx if padding_idx is not None else 0
        # self.weights = UnicodeCharactersEmbedding.get_embedding(
        #     init_size, embedding_dim, vocab, self.char_rand_embedding, padding_idx
        # )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    ########### UNICODE VERSION #############
    # @staticmethod
    # def get_embedding(
    #         num_embeddings: int, embedding_dim: int, vocab: dict, padding_idx: Optional[int] = None
    # ):
    #     """Build Unicode-UTF-8 characters embeddings.
    #     All tokens in vocab assumed to be maximum length of embedding_dim/32
    #
    #     """
    #     print("Unicode emb addition")
    #     emb_matrix = torch.zeros(len(vocab.symbols), embedding_dim)
    #     for i, token in enumerate(vocab.symbols):
    #         if i < 4:  # for unique tokens <s>, <pad>, </s>, <unk>
    #             emb_matrix[i, :] = torch.zeros(embedding_dim)
    #             continue
    #         # token_uni_emb = []
    #         for j, c in enumerate(token):
    #             # char_enc = repr(c).encode('utf-8')  # get the bytes data, '\xd7\x90' for exmaple repr('a')='a'
    #             char_enc = c.encode('utf-8')  # get the bytes data, '\xd7\x90' for exmaple
    #             char_bits = bin(int.from_bytes(char_enc, byteorder='big'))[2:]  # string of bits
    #             if j >= embedding_dim/32:  # no more room for more chars in this emb
    #                 continue
    #             for k, bit in enumerate(char_bits):
    #                 index = -(k + 32 * j) - 1
    #                 if bit == '1':
    #                     emb_matrix[i, index] = 1
    #                 else:
    #                     emb_matrix[i, index] = -1
    #             # char_bits_ints = [1 if x == '1' else -1 for x in char_bits]  # list of int for each bit
    #             # token_uni_emb += char_bits_ints
    #         # emb_matrix[i, -len(token_uni_emb):] = torch.tensor(list(reversed(token_uni_emb)))  # use the last part of the vector, not to intersect with positional emb
    #     return emb_matrix

    ########## VOCAB VERSION #############
    @staticmethod
    def get_embedding(
            num_embeddings: int, embedding_dim: int, vocab: dict, char_rand_embedding=None, input_tokens=None,
            tokens_chars_idx=None,
            padding_idx: Optional[int] = None
    ):
        """Build Vocab characters embeddings.

        """

        # emb_matrix = torch.nn.Parameter(torch.zeros(len(vocab.symbols), embedding_dim))
        # emb_matrix = nn.Embedding(len(vocab.symbols), embedding_dim, _weight=torch.zeros(len(vocab.symbols), embedding_dim))
        # emb_matrix = torch.zeros(len(vocab.symbols), embedding_dim).cuda()

        # if input_tokens is not None:
        #     input_strings = [vocab.symbols[i_token] for i_token in input_tokens]
        # else:
        #     input_tokens = [i for i in range(len(vocab.symbols))]
        #     input_strings = vocab.symbols

        # emb_matrix = torch.zeros(len(input_tokens), embedding_dim).cuda()

        # chars_in_vocab = UnicodeCharactersEmbedding.get_chars_in_vocab(vocab.symbols)
        # bits_need_per_char = len(bin(len(chars_in_vocab))) - 2
        # print("bits_need_per_char", bits_need_per_char)

        if input_tokens is not None:
            chars_idx = tokens_chars_idx[input_tokens.view(-1)].cuda()
            emb_matrix = F.embedding(chars_idx, char_rand_embedding.weight).view(len(input_tokens), -1)
            return emb_matrix
        else:
            raise Exception("Need to uncomment some code here")

        # for i, token in enumerate(input_tokens):
        #     if i < 4:  # for unique tokens <s>, <pad>, </s>, <unk>
        #         # emb_matrix[i, :] = emb_matrix[i, :] + torch.zeros(embedding_dim)
        #         continue
        #     token = expi(input_strings[i], double_char, without_shtrudel, word_marking, lang_marking)
        #     if rand_embedding:
        #         token_emb = None
        #
        #     for j, c in enumerate(input_strings[i]):
        #         if is_j_out_of_bounds(j):
        #             continue
        #         if rand_embedding:
        #             # if chars_in_vocab.index(c) == 9 and len(token) == 1:
        #             #     print("chars_in_vocab.index(c) == ", chars_in_vocab.index(c))
        #             #     print("token number 6:", token)
        #             #     print("char_emb(chars_in_vocab.index(c))", char_emb(chars_in_vocab.index(c)))
        #             if token_emb is None:
        #                 token_emb = [char_emb(chars_in_vocab.index(c))]  # first char
        #             else:
        #                 # token_emb = torch.cat((token_emb, char_emb(chars_in_vocab.index(c))), dim=0) # concat chars
        #                 token_emb.append(char_emb(chars_in_vocab.index(c)))
        #             continue
        #
        #         char_bits = bin(chars_in_vocab.index(c))[2:]
        #         if geresh_padding:
        #             begin_pad = "2" * bits_need_per_char
        #             mid_pad = "0" * (bits_need_per_char - len(char_bits))
        #             end_pad = "3" * bits_need_per_char
        #             char_bits = begin_pad + mid_pad + char_bits + end_pad
        #         if super_geresh_padding:
        #             begin_pad = "00100111"
        #             mid_pad = "0" * (bits_need_per_char - len(char_bits))
        #             end_pad = "00100111"
        #             char_bits = begin_pad + mid_pad + char_bits + end_pad
        #
        #         for k, bit in enumerate(char_bits):
        #             index = get_index(j, k, bits_need_per_char)
        #
        #             if bit == '1':
        #                 emb_matrix[i, index] = 1
        #             elif bit == '0':
        #                 emb_matrix[i, index] = -1
        #             elif bit == '2':
        #                 emb_matrix[i, index] = 0.5
        #             elif bit == '3':
        #                 emb_matrix[i, index] = -0.5
        #             else:
        #                 print("token: ", token)
        #                 print("char_bits: ", char_bits)
        #                 print("bit: ", bit)
        #                 raise Exception("Something went wrong, in char_bits should have only bits of 0,1. for pad 2,3,")
        #     if rand_embedding:  # place concatenated chars in emb_matrix
        #         token_emb = torch.cat(token_emb, dim=0)
        #         emb_matrix[i, -token_emb.shape[0]:] += token_emb.unsqueeze(0).fliplr().squeeze()
        #         # token_emb.append(torch.zeros(embedding_dim - len(token_emb) * token_emb[0].shape[0]))
        #         # token_emb = torch.cat(token_emb, dim=0)  # concat all chars
        #         # emb_matrix[i] = token_emb.unsqueeze(0).fliplr().squeeze()
        #         # emb_list.append(token_emb.unsqueeze(0).fliplr())

        # return emb_matrix.cuda()
        # return torch.cat(emb_list, dim=0)
        # return nn.Embedding(len(vocab.symbols), embedding_dim, _weight=emb_matrix)

    ########### SIN-COS VERSION #############
    # @staticmethod
    # def get_embedding(
    #         num_embeddings: int, embedding_dim: int, vocab: dict, padding_idx: Optional[int] = None
    # ):
    #     """Build sinusoidal embeddings.
    #
    #     This matches the implementation in tensor2tensor, but differs slightly
    #     from the description in Section 3.5 of "Attention Is All You Need".
    #     """
    #
    #     def get_chars_in_vocab(vocab):
    #         chars_in_vocab = set()
    #         for token in vocab.symbols:
    #             for c in token:
    #                 if c not in chars_in_vocab:
    #                     chars_in_vocab.add(str(c))
    #         return sorted(list(chars_in_vocab))
    #
    #     def get_char_vectors(embedding_dim: int, chars_in_vocab: dict):
    #         num_embeddings = len(chars_in_vocab)
    #
    #         # full_dim_vectors = torch.zeros(num_embeddings, embedding_dim)
    #         # embedding_dim = embedding_dim // 2
    #
    #         half_dim = embedding_dim // 2
    #         emb = math.log(10000) / (half_dim - 1)
    #         emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    #         emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
    #             1
    #         ) * emb.unsqueeze(0)
    #         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
    #             num_embeddings, -1
    #         )
    #         if embedding_dim % 2 == 1:
    #             # zero pad
    #             emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    #         if padding_idx is not None:
    #             emb[padding_idx, :] = 0
    #
    #         # full_dim_vectors[:, embedding_dim:] = emb
    #         # emb = full_dim_vectors
    #
    #         return torch.fliplr(emb)  # flip left to right so char emb won't collide with positional emb
    #
    #     print("Sin emb addition")
    #     chars_in_vocab = get_chars_in_vocab(vocab)
    #     char_vectors = get_char_vectors(embedding_dim, chars_in_vocab)
    #     emb_matrix = torch.zeros(len(vocab.symbols), embedding_dim)
    #     for i, token in enumerate(vocab.symbols):
    #         if i < 4:  # for unique tokens <s>, <pad>, </s>, <unk>
    #             emb_matrix[i, :] = torch.zeros(embedding_dim)
    #             continue
    #         for c in token:
    #             t = chars_in_vocab.index(c)
    #             emb_matrix[i] += char_vectors[t]
    #     return emb_matrix

    ########### PRE_TRAINED LOAD VERSION #############
    # @staticmethod
    # def get_embedding(
    #         num_embeddings: int, embedding_dim: int, vocab: dict, padding_idx: Optional[int] = None
    # ):
    #     """Build characters embeddings from pre-trained LM.
    #
    #     """
    #     name_to_load = '/home/olab/itayitzhak/bpeplus/fairseq/checkpoints/transformer_tokens2char_de-en_opt2_medium_model_3_0.3/checkpoint_best.pt'
    #     lm_model = torch.load(name_to_load)
    #     print("Pre-trained emb addition")
    #     if len(vocab.symbols) == lm_model['model']['decoder.embed_tokens.weight'].shape[0]:  # load only for encoder embeddings
    #         print("Loading embeddings, dict size:" + str(len(vocab)))
    #         print("From file:" + name_to_load)
    #         emb = lm_model['model']['decoder.embed_tokens.weight']
    #     else:
    #         print("NOT Loading embeddings, dict size:" + str(len(vocab)))
    #         print("Tried to load emb matrix sized:", lm_model['model']['decoder.embed_tokens.weight'].shape)
    #         print("NOT loading from file:" + name_to_load)
    #
    #     return emb

    def forward(
            self,
            input,
            incremental_state: Optional[Any] = None,
            timestep: Optional[Tensor] = None,
            positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        # if self.weights is None or max_pos > self.weights.size(0):
        #     # recompute/expand embeddings if needed
        #     self.weights = UnicodeCharactersEmbedding.get_embedding(
        #         max_pos, self.embedding_dim, self.vocab, self.char_rand_embedding, self.padding_idx
        #     )

        # if self.weights is None or max_pos > self.weights.weight.size(0):
        if False and (self.weights is None or max_pos > self.weights.size(0)):
            # recompute/expand embeddings if needed
            self.weights = UnicodeCharactersEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.vocab, self.char_rand_embedding, self.padding_idx
            )
            self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] if timestep is not None else seq_len - 1
            if self.onnx_trace:
                if False:
                    flat_embeddings = self.weights.detach().index_select(0, input[:, pos].view(-1))
                    embedding_shape = torch.cat(
                        (bsz.view(1), torch.tensor([1], dtype=torch.long), torch.tensor([-1], dtype=torch.long))
                    )
                    embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                        flat_embeddings, embedding_shape
                    )
                    return embeddings
                else:
                    flat_embeddings = UnicodeCharactersEmbedding.get_embedding(
                        max_pos, self.embedding_dim, self.vocab, self.char_rand_embedding, input[:, pos].view(-1),
                        self.tokens_chars_idx,
                        self.padding_idx
                    ).detach()
                    embedding_shape = torch.cat(
                        (bsz.view(1), torch.tensor([1], dtype=torch.long), torch.tensor([-1], dtype=torch.long))
                    )
                    embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                        flat_embeddings, embedding_shape
                    )
                    return embeddings

            if False:
                return (
                    self.weights.index_select(0, input[:, pos].view(-1))
                        .view(bsz, 1, -1)
                        .detach()
                )
            else:
                return (
                    UnicodeCharactersEmbedding.get_embedding(
                        max_pos, self.embedding_dim, self.vocab, self.char_rand_embedding, input[:, pos].view(-1),
                        self.tokens_chars_idx,
                        self.padding_idx
                    ).view(bsz, 1, -1).detach()
                )

        if self.onnx_trace:
            if False:
                # flat_embeddings = self.weights.detach().index_select(0, input.view(-1))
                flat_embeddings = UnicodeCharactersEmbedding.get_embedding(
                    max_pos, self.embedding_dim, self.vocab, self.char_rand_embedding, input.view(-1),
                    self.tokens_chars_idx,
                    self.padding_idx
                )
            else:
                # flat_embeddings = self.weights.index_select(0, input.view(-1))
                flat_embeddings = UnicodeCharactersEmbedding.get_embedding(
                    max_pos, self.embedding_dim, self.vocab, self.char_rand_embedding, input.view(-1),
                    self.tokens_chars_idx,
                    self.padding_idx
                )
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings

        if False:
            # return (
            #     self.weights.index_select(0, input.view(-1))
            #         .view(bsz, seq_len, -1)
            #         .detach()
            # )
            return UnicodeCharactersEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.vocab, self.char_rand_embedding, input.view(-1),
                self.tokens_chars_idx,
                self.padding_idx
            ).view(bsz, seq_len, -1).detach()
        else:
            # when get_emb returns an emb matrix
            # return (
            #     self.weights(input.view(-1))
            #         .view(bsz, seq_len, -1)
            # )
            # return (
            #     self.weights.index_select(0, input.view(-1))
            #         .view(bsz, seq_len, -1)
            # )
            # return (
            #     F.embedding(input.view(-1), self.weights).view(bsz, seq_len, -1)
            # )
            return UnicodeCharactersEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.vocab, self.char_rand_embedding, input.view(-1),
                self.tokens_chars_idx,
                self.padding_idx
            ).view(bsz, seq_len, -1)
