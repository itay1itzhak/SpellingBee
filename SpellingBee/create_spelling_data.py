import torch
import numpy as np
import os
import argparse
import itertools
from tqdm.auto import tqdm

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()


def preprocess_dic(input_dic):
    input_dic = open(input_dic)
    input_dic = [x.split(' ')[0] for x in input_dic]
    return input_dic


def normalize(token_index, cur_dict, fname):
    '''make token varations in RoBERTa's dict into the same token '''

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    if 'roberta' in fname:
        token = cur_dict[token_index]
        token = token.replace('Ġ', '').lower()
    if 'gpt2' in fname:
        token = cur_dict[token_index]
        token = token.replace('Ġ', '').lower()
    if 'AraBERT' in fname:
        token = cur_dict[token_index]
        token = token.replace('##', '').lower()
        return stemmer.stem(token)
    if 'GloVe' in fname:
        return cur_dict[token_index]

    if len(token) == 0:
        return token
    token_pos = nltk.pos_tag([token])[0][1]
    token_pos = get_wordnet_pos(token_pos)
    if token_pos is '':
        return lemmatizer.lemmatize(token)
    else:
        return lemmatizer.lemmatize(token, pos=token_pos)


def get_emb_matrix(fname):
    ''' Load the model embeddings matrix and return it'''
    if 'roberta.base' in fname:
        roberta = torch.load('roberta.base/model.pt')
        loaded_emb_matrix = roberta['model']['decoder.sentence_encoder.embed_tokens.weight'][4:-4, :]
    if 'roberta.large' in fname:
        roberta = torch.load('roberta.large/model.pt')
        loaded_emb_matrix = roberta['model']['decoder.sentence_encoder.embed_tokens.weight'][4:-4, :]
    if 'gpt2' in fname:
        loaded_emb_matrix = torch.load('gpt2/gpt2-medium-emb.pt')
    if 'AraBERT' in fname:
        loaded_emb_matrix = torch.load('AraBERT/bert-large-arabertv02_emb.pt')
        loaded_emb_matrix = loaded_emb_matrix[5:60001, :]  # token2char file does not contain speical tokens
    if 'GloVe' in fname:
        loaded_emb_matrix = torch.load('GloVe/GloVe.6B.300d.emb.pt')
        loaded_emb_matrix = loaded_emb_matrix[:50000, :]
    return loaded_emb_matrix


def get_cosine_sim(emb_for_cosine):
    emb_nor = torch.linalg.norm(emb_for_cosine, ord=2, dim=-1)
    emb_nor = emb_nor.unsqueeze(1).expand(emb_for_cosine.size())
    sims = emb_for_cosine / emb_nor
    sims = sims @ sims.T
    return sims


def get_top_k_similar(emb, test_set, k):
    sims = get_cosine_sim(emb)
    top_k = torch.topk(sims, k=k, dim=-1)[1]
    return top_k[test_set, :]


def is_whole_word(token_index, cur_dict, fname):
    if 'roberta' in fname:
        return 'Ġ' in cur_dict[token_index]
    return None


def get_similarity_split(fname, cur_dict, file_len, k, test_indices, whole_words_only, with_lemma=True, verbose=False):
    ''' Returns a data split of random words in tests, and words that are not similar in train'''
    emb_matrix = get_emb_matrix(fname)

    test_normalized_tokens = set()
    if with_lemma:
        for token_index in test_indices:
            test_normalized_tokens.add(normalize(token_index, cur_dict, fname))

    most_similar_words_to_test_set = get_top_k_similar(emb_matrix, test_indices, k)

    if verbose:
        print("test_normalized_tokens", test_normalized_tokens)
        print("most_similar_words_to_test_set", most_similar_words_to_test_set)
    count_normal_in_test = 0
    count_normal_in_test_and_not_similar = 0

    train_indices = []
    for i in range(file_len):
        if i not in most_similar_words_to_test_set and normalize(i, cur_dict, fname) not in test_normalized_tokens:
            if whole_words_only and not is_whole_word(i, cur_dict, fname):
                continue
            train_indices.append(i)
        if verbose:
            if normalize(i, cur_dict, fname) in test_normalized_tokens:
                print("This word normalized version is in test:", end=' |')
                print(cur_dict[i], end=' | normalize:')
                print(normalize(i, cur_dict, fname))
                count_normal_in_test += 1
            if i not in most_similar_words_to_test_set and normalize(i, cur_dict, fname) in test_normalized_tokens:
                print("Not 20 closest, but it's normalized version is in test:", end=' |')
                print(i, end=' |')
                print(cur_dict[i], end=' | normalize:')
                print(normalize(i, cur_dict, fname))
                count_normal_in_test_and_not_similar += 1
            print("count_normal_in_test:", count_normal_in_test)
            print("count_normal_in_test_and_not_similar:", count_normal_in_test_and_not_similar)

    return np.random.permutation(train_indices)


def get_file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def get_random_split(fname, cur_dict, file_len, test_indices, whole_words_only):
    train_indices = []
    if whole_words_only:
        indices_to_choose_from = [i for i in range(file_len) if is_whole_word(i, cur_dict, fname)]
    else:
        indices_to_choose_from = range(file_len)

    for i in indices_to_choose_from:
        if i not in test_indices:
            train_indices.append(i)
    return np.random.permutation(train_indices)


def get_split_indicies(fname, cur_dict, file_len, split_state, train_size, whole_words_only, test_indices=None):
    if split_state == 'random_split':
        return get_random_split(fname, cur_dict, file_len, test_indices, whole_words_only)[:train_size], None
    if split_state == 'similarity_split':
        return get_similarity_split(fname, cur_dict, file_len, args.top_k_sim_remove, test_indices, whole_words_only,
                                    with_lemma=False)[
               :train_size], None
    if split_state == 'lemma_similarity_split':
        return get_similarity_split(fname, cur_dict, file_len, args.top_k_sim_remove, test_indices, whole_words_only,
                                    with_lemma=True)[
               :train_size], None


def create_train_val_test_files(input_fname, cur_dict, file_len, split_state, train_size, output_prefix, test_indices,
                                whole_words_only):
    train_file = open(output_prefix + '.train', "w", encoding='utf8')
    valid_file = open(output_prefix + '.valid', "w", encoding='utf8')
    test_file = open(output_prefix + '.test', "w", encoding='utf8')

    train_indices, _ = get_split_indicies(input_fname, cur_dict, file_len, split_state, train_size, whole_words_only,
                                          test_indices)

    with open(input_fname, 'r', encoding='utf8') as input_f:
        for i, line in enumerate(input_f.readlines()):
            if i in train_indices:
                train_file.write(line)
            elif i in test_indices:
                valid_file.write(line)
                test_file.write(line)
    train_file.close()
    valid_file.close()
    test_file.close()


def create_data_and_preprocess(args, seed, pb, cur_dict, dict_path=None, whole_words_only=False):
    input_fname = args.data_dir + 'tokens2char.' + args.model + '.txt'
    file_len = get_file_len(input_fname)

    if args.divide_test_by_freq:
        freq_list = [range(0, 10000), range(10000, 20000), range(20000, 30000), range(30000, 40000),
                     range(40000, 50000)]
    else:
        freq_list = [range(file_len)]

    for freq_split in freq_list:
        if whole_words_only:
            indices_to_choose_from = [i for i in freq_split if is_whole_word(i, cur_dict, input_fname)]
        else:
            indices_to_choose_from = freq_split

        test_indices = np.random.choice(indices_to_choose_from, int(args.test_size),
                                        replace=False)

        for split_state, train_size in list(itertools.product(args.splits, args.train_sizes)):
            output_prefix = input_fname + '.' + f'{seed}_{split_state}_train_size_{train_size}'
            if args.divide_test_by_freq:
                freq_name = str(freq_split).replace(' ', '').replace('(', '_').replace(')', '_').replace(',', '_')
                output_prefix += f"_freq_{freq_name}"
            if whole_words_only:
                output_prefix += "_whole_words"
            create_train_val_test_files(input_fname, cur_dict, file_len, split_state, train_size, output_prefix,
                                        test_indices,
                                        whole_words_only)
            pb.update(1)


def run_main(args):
    if 'roberta' in args.model:
        dict_path = f'{args.model}/real_dict.txt'
    else:
        dict_path = f'{args.model}/dict.txt'
    cur_dict = preprocess_dic(dict_path)
    if 'glove' in args.model:
        dict_path = dict_path[:50000]

    if 'arabert' in args.model or 'AraBERT' in args.model:
        from farasa.stemmer import FarasaStemmer
        global stemmer
        stemmer = FarasaStemmer(interactive=True)

    if not os.path.exists(f"{args.data_dir}"):
        os.makedirs(f"{args.data_dir}")

    args.seeds = [int(x) for x in args.seeds.split(',')]
    args.splits = args.splits.split(',')
    args.train_sizes = [int(x) for x in args.train_sizes.split(',')]


    pb = tqdm(total=len(args.seeds))
    for seed in args.seeds:
        np.random.seed(int(seed))
        create_data_and_preprocess(args, seed, pb, cur_dict=cur_dict, dict_path=dict_path,
                                   whole_words_only=args.whole_words_only)
    pb.close()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", type=str, default="spelling_data/", help=""
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="model can be roberta.base, roberta.large, gpt2, arabert or glove.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2",
        help="The original paper used 0,1,2,3,4,5,5,6,7,8,9",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="random_split,similarity_split,lemma_similarity_split",
        help="None filter is random_split, Similarity filter is similarity_split, and Lemma filter is lemma_similarity_split.",
    )
    parser.add_argument(
        "--train_sizes",
        type=str,
        default="32000",
        help="The original paper used 32000.",
    )
    parser.add_argument(
        "--divide_test_by_freq",
        type=bool,
        default=False,
        help="This allows to take tests tokens according the frequencies in roberta training, as was done for the appendix in the paper.",
    )

    parser.add_argument(
        "--top_k_sim_remove",
        type=int,
        default=20,
        help="For each token in tests, how many top k similar tokens to remove from train, used k=20 in the paper.",
    )

    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Size of the test set, used 1000 for paper results.",
    )

    parser.add_argument(
        "--whole_words_only",
        type=bool,
        default=False,
        help="If to test whole words only, used for GloVe and CharacterBERT experiments.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    run_main(args)
