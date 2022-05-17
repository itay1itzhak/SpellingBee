import itertools
import os

# folder = '/home/olab/itayitzhak/bpeplus/fairseq/data-bin/iwslt14.tokenized.ar-en/'
# INPUT = folder + 'dict.en.txt'
# OUTPUT = folder + 'tokens2char.ar.opt2.reversed.only.txt'
is_roberta = False
is_gpt2 = False
is_AraBERT = False
is_glove = True

lang = "ar"
folder = '/home/olab/itayitzhak/bpeplus/fairseq/data-bin/'


def get_sorted_roberta_dict():
    import numpy as np
    sorted_roberta_dict = []
    with open("/home/olab/itayitzhak/bpeplus/roberta.base/roberta_dict.txt", 'r') as sorted_roberta_dict_file:
        for line in sorted_roberta_dict_file.readlines():
            sorted_roberta_dict.append(line.split(' ')[0])

    return sorted_roberta_dict

if is_roberta:
    INPUT = '/home/olab/itayitzhak/bpeplus/roberta.base/dict.txt'
    OUTPUT = folder + 'spelling_data/' + 'tokens2char.roberta.base.txt'
    REAL_DICT = '/home/olab/itayitzhak/bpeplus/roberta.base/real_dict2.txt'
    import torch

    print("Loading RoBERTa...")
    from fairseq.models.roberta import RobertaModel

    roberta_speical_load = RobertaModel.from_pretrained('/home/olab/itayitzhak/bpeplus/roberta.base',
                                                        checkpoint_file='model.pt')
    print("Done Loading RoBERTa.")

    sorted_roberta_dict = get_sorted_roberta_dict()

elif is_gpt2:
    INPUT = '/home/olab/itayitzhak/bpeplus/gpt2/dict.txt'
    OUTPUT = folder + 'spelling_data/organized_tests/' + 'tokens2char.gpt2.medium.txt'
elif is_AraBERT:
    INPUT = '/home/olab/itayitzhak/bpeplus/AraBERT/dict.txt'
    OUTPUT = folder + 'spelling_data/organized_tests/' + 'tokens2char.AraBERT.txt'
elif is_glove:
    INPUT = '/home/olab/itayitzhak/bpeplus/GloVe/glove.6B.300d.txt'
    DICT_OUTPUT = '/home/olab/itayitzhak/bpeplus/GloVe/' + 'dict.txt'
    OUTPUT = folder + 'spelling_data/organized_tests/' + 'tokens2char.GloVe.txt'
    sorted_roberta_dict = get_sorted_roberta_dict()
else:
    INPUT = folder + f'iwslt14.tokenized.{lang}-en/' + 'dict.en.txt'
    OUTPUT = folder + 'spelling_data/organized_tests/' + f'tokens2char.{lang}.with@.txt'


# opt_0 = goo@@ g o o
# opt_1 = goo@@ g@@ o@@ o@@
# opt_2 = goo@@ !!g !!o !!o !!@@
# opt_3 = goo@@ g o o @@

def create_pretrain_data():
    def write_token_per_dict(dict_name):
        def get_all_permutations(l):
            idx = [i for i in range(l)]  # normal order

            # result = list(itertools.permutations(idx)) # all permutations
            result = [idx]
            # result = [idx, [j for j in range(l - 1, -1, -1)]]  # normal and reverse
            # result = [[j for j in range(l - 1, -1, -1)]] # reverse

            return result

        def write_chars(token, chars_idx):
            # for i in chars_idx:
            if is_AraBERT:
                token = token.replace('##', '#')
            for c in token:
                if is_glove:
                    all_chars.add(c)
                if c is '@' and not is_roberta and not is_gpt2 and not is_AraBERT:
                    tokens2char_file.write(' @')
                    break
                else:
                    tokens2char_file.write(' ' + c)
                    # tokens2char_file.write(' ' + c + '!!@@')

            tokens2char_file.write(' \n')

        # with open(dict_name, 'r', encoding='utf8') as input_f:
        with open(dict_name, 'r') as input_f:
            for i, line in enumerate(input_f.readlines()):
                token = line.split(' ')[0]
                if is_roberta:
                    print("line: ", line)
                    # if token == '1' or 'madeupword' in token:
                    if 'madeupword' in token:
                        continue
                    # token = roberta_speical_load.decode(torch.tensor([int(token)]))
                    token = sorted_roberta_dict[int(token)]
                    # direction = '!!normal '
                    # new_dict.write(token.replace(' ', '@'))
                    new_dict.write(token)
                    new_dict.write(' ' + line.split(' ')[1])
                if is_AraBERT:
                    if '[UNUSED_' in token or '[PAD]' == token or '[UNK]' == token or '[CLS]' == token or '[SEP]' == token or '[MASK]' == token:
                        continue
                if is_glove:
                    if len(token) == 1:
                        chars_already_exists.add(token)
                        dict_file.write(token + f" {i}\n")
                        continue
                    if i > 50000:
                    #if token not in sorted_roberta_dict:
                        continue
                    dict_file.write(token+f" {i}\n")

                for chars_idx in get_all_permutations(len(token)):
                    # tokens2char_file.write(direction)
                    tokens2char_file.write(token)
                    write_chars(token, chars_idx)
                    # direction = '!!reversed '

    tokens2char_file = open(OUTPUT, 'w+')  # , encoding='utf8')
    if is_roberta:
        new_dict = open(REAL_DICT, 'w+')
    if is_glove:
        dict_file = open(DICT_OUTPUT, 'w+')  # , encoding='utf8')
        all_chars = set()
        chars_already_exists = set()

    write_token_per_dict(INPUT)

    tokens2char_file.close()
    if is_roberta:
        new_dict.close()
    if is_glove:
        for i, c in enumerate(all_chars):
            if c not in chars_already_exists:
                dict_file.write(c + f" {i}\n")


if __name__ == "__main__":
    create_pretrain_data()
