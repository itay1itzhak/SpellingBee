from sacrebleu.metrics import CHRF
import argparse
import numpy as np
import re
import itertools
import os
from tqdm.auto import tqdm


def get_edit_distance(s1, s2):
    dist = 0
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            dist += 1
    dist += abs(len(s1) - len(s1))

    return dist


def get_char_f1(token_from_chars, token):
    precision = 0
    recall = 0

    if len(token_from_chars) == 0:
        return 0

    for c in token_from_chars:
        if c in token:
            precision += 1
    for c in token:
        if c in token_from_chars:
            recall += 1

    precision /= len(token_from_chars)
    recall /= len(token)

    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)


def levenshtein_ratio_and_distance(s, t, ratio_calc=True):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0  # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row - 1][col] + 1,  # Cost of deletions
                                     distance[row][col - 1] + 1,  # Cost of insertions
                                     distance[row - 1][col - 1] + cost)  # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s) + len(t)) - distance[len(s)][len(t)]) / (len(s) + len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return distance[len(s)][len(t)]


class chrfArg():
    def __init__(self):
        self.chrf_whitespace = False
        self.chrf_order = 6
        self.chrf_beta = 2


def is_token_in_test_group(fname, cur_dict, token, test_group):
    if test_group is None or test_group == "":  # no test groups
        return True
    if 'roberta' not in fname:
        raise Exception("Sorry, Analysis only for roberta")

    if type(test_group) == type(range(0, 10000)):  # freq
        return cur_dict.index(token) in test_group
    if type(test_group) == type(1):  # length
        if test_group > 10:  # bucket token legnth
            return len(token) > 10
        return len(token) == test_group

    if test_group == 'whole_words':  # whole/sub words
        return 'Ġ' in token
    if test_group == 'sub_words':  # whole/sub words
        return 'Ġ' not in token

    if '_sub_words' in test_group:  # length+whole/sub words
        return 'Ġ' not in token and len(token) == int(test_group[:2])
    if '_whole_words' in test_group:  # length+whole/sub words
        return 'Ġ' in token and len(token) == int(test_group[:2])


def get_accuracy(fname, cur_dict, exact_match=True, test_group=None):
    def update_token(token, total_num, num_correct, num_correct_num_of_chars, edit_distance, char_f1, levenshtein,
                     chrf):
        total_num += 1
        if exact_match and token == token_from_chars and len(token) > 0:
            num_correct += 1
        if len(token) == len(token_from_chars):
            num_correct_num_of_chars += 1

        edit_distance += get_edit_distance(token_from_chars, token)
        char_f1 += get_char_f1(token_from_chars, token)
        levenshtein += levenshtein_ratio_and_distance(token_from_chars, token, ratio_calc=True)
        # if len(token_from_chars) > 0:
        chrf += float(chrf_scorer.corpus_score([token_from_chars], [[token]]).__str__()[8:])
        return total_num, num_correct, num_correct_num_of_chars, edit_distance, char_f1, levenshtein, chrf

    total_num, num_correct, num_correct_num_of_chars, edit_distance, char_f1, levenshtein, chrf = 0, 0, 0, 0, 0, 0, 0
    chrf_scorer = CHRF(chrfArg())

    with open(fname, 'r') as f:
        for line in f.readlines():
            if line[0] == 'H':
                line = line.replace('\n', '').split('\t')[2].split(' ')
                token = line[0].strip()
                token_from_chars = ''.join(line[1:])
                if not is_token_in_test_group(fname, cur_dict, token, test_group):
                    continue
                total_num, num_correct, num_correct_num_of_chars, edit_distance, char_f1, levenshtein, chrf = update_token(
                    token, total_num, num_correct, num_correct_num_of_chars, edit_distance, char_f1, levenshtein,
                    chrf)

            if 'Generate test with beam=1: BLEU4' in line:
                bleu_idx = re.search('BLEU4', line).span()[1] + 3
                try:
                    bleu_score = float((line[bleu_idx:bleu_idx + 5]))
                except:
                    bleu_score = float((line[bleu_idx:bleu_idx + 4]))
    if total_num == 0:
        print("\nFile is empty.")
        print("File name: "+fname.split('/')[7])
    return num_correct / total_num, bleu_score, num_correct_num_of_chars / total_num, edit_distance / total_num, char_f1 / total_num, levenshtein / total_num, chrf / total_num, total_num


def create_results_table(args, dict_path):
    if args.whole_words:
        whole_words = "_whole_words"
    else:
        whole_words = ""

    if args.analysis_mode is None:
        test_group_options = [None]
        results_file_path = f"{args.results_dir}{args.model}{whole_words}_results.csv"
    else:
        if args.analysis_mode == 'length_bucket':
            test_group_options = [i for i in range(1, 12)]
        if args.analysis_mode == 'whole_vs_sub_words':
            test_group_options = ['whole_words', 'sub_words']
        if args.analysis_mode == 'length_and_subword':
            test_group_options = [f'{i:02}' + '_whole_words' for i in range(1, 19)]
            test_group_options += [f'{i:02}' + '_sub_words' for i in range(1, 19)]

        results_file_path = f"{args.results_dir}{args.model}{whole_words}{args.analysis_mode}_results.csv"

    if args.checkpoints_dir is None:
        cp_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/checkpoints'
    else:
        cp_dir = args.checkpoints_dir

    results_file = open(results_file_path, 'w+')
    results_file.write(
        "Filter,Is_pretrained_embeddings,Train Size,EM_score,Bleu_score,Edit_dist,Levinstein,f_1,chrf,Correct Length, Test group\n")

    if args.random_init_emb_prefix is not None:
        trained_prefix_list = [args.trained_emb_prefix, args.random_init_emb_prefix]
    else:
        trained_prefix_list = [args.trained_emb_prefix]
    all_models = list(
        itertools.product(trained_prefix_list, args.splits, args.train_sizes, args.freq_list, test_group_options))
    pb = tqdm(total=len(args.seeds) * len(all_models))
    for is_pretrained, split_state, train_size, freq_name, test_group in all_models:
        sum_curr_model_em, sum_curr_model_bleu, sum_curr_model_len, sum_curr_model_edit_distance, sum_curr_model_f_1, sum_curr_model_levinstein, sum_curr_model_chrf, num_calculated = 0, 0, 0, 0, 0, 0, 0, 0

        for seed in args.seeds:
            model_to_eval = f'{cp_dir}/{is_pretrained}_{args.model}_{seed}_{split_state}_train_size_{train_size}{freq_name}{whole_words}'

            acc, bleu, correct_len, edit_distance, f_1, levinstein, chrf, _ = get_accuracy(
                model_to_eval + "/generate-test.txt", dict_path, test_group=test_group)

            sum_curr_model_em += acc
            sum_curr_model_bleu += bleu
            sum_curr_model_len += correct_len
            num_calculated += 1
            sum_curr_model_edit_distance += edit_distance
            sum_curr_model_f_1 += f_1
            sum_curr_model_levinstein += levinstein
            sum_curr_model_chrf += chrf
            # break
        if num_calculated > 0:
            em_score = "{:.4}".format(sum_curr_model_em / num_calculated)
            bleu_score = "{}".format(sum_curr_model_bleu / num_calculated)
            edit_distance_score = "{:.4}".format(sum_curr_model_edit_distance / num_calculated)
            f_1_score = "{:.2}".format(sum_curr_model_f_1 / num_calculated)
            levinstein_score = "{:.4}".format(sum_curr_model_levinstein / num_calculated)
            chrf_score = "{:.4}".format(sum_curr_model_chrf / num_calculated)

            correct_len_score = "{:.2}".format(sum_curr_model_len / num_calculated)

            results_file.write(
                f"{split_state},{is_pretrained},{train_size},{em_score},{bleu_score},{edit_distance_score},{levinstein_score},{f_1_score},{chrf_score},{correct_len_score},{test_group}\n")
            pb.update(1)

    results_file.close()
    pb.close()


def run_main(args):
    if 'roberta' in args.model:
        dict_path = f'{args.model}/real_dict.txt'
    else:
        dict_path = f'{args.model}/dict.txt'

    if not os.path.exists(f"{args.results_dir}"):
        os.makedirs(f"{args.results_dir}")

    args.seeds = [int(x) for x in args.seeds.split(',')]
    args.splits = args.splits.split(',')
    args.train_sizes = [int(x) for x in args.train_sizes.split(',')]
    args.freq_list = args.freq_list.split(',')

    create_results_table(args, dict_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", type=str, default="spelling_data/", help="The dir for all the spelling data."
    )
    parser.add_argument(
        "--checkpoints_dir", type=str, default=None, help="default is the checkpoint folder in the parent dir of working directory which is assumed as fairseq."
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/", help="The dir to save the results to."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="roberta.base",
        help="model can be roberta.base, roberta.large, gpt2, arabert or glove.",
    )

    parser.add_argument(
        "--trained_emb_prefix",
        type=str,
        default="pretrained",
        help="The first word in the model dir, indicates the probe trained with pretrained embeddings.",
    )

    parser.add_argument(
        "--random_init_emb_prefix",
        type=str,
        default=None,
        help="The first word in the model dir, indicates the probe was trained with randomized init embeddings.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="1",
        help="The original paper used 0,1,2,3,4,5,5,6,7,8,9",
    )
    parser.add_argument(
        "--splits",
        type=str,
        #default="random_split,similarity_split,lemma_similarity_split",
        default="similarity_split",
        help="None filter is random_split, Similarity filter is similarity_split, and Lemma filter is lemma_similarity_split.",
    )
    parser.add_argument(
        "--train_sizes",
        type=str,
        default="32000",
        help="The original paper used 32000.",
    )
    parser.add_argument(
        "--freq_list",
        type=str,
        default="",
        help="This allows to split according the frequencies in roberta, as was done for the appendix in the paper.",
    )

    parser.add_argument(
        "--whole_words",
        type=bool,
        default=False,
        help="If to test probe trained on whole words only, used for comparison with GloVe and CharacterBERT experiments.",
    )

    parser.add_argument(
        "--analysis_mode",
        type=str,
        default=None,
        help="If set to true, do not produce regular spelling result, but advance analysis - results for length ('length_bucket')  or whole words vs. subwords ('whole_vs_sub_words')",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    run_main(args)
