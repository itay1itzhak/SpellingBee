import os
import argparse
import torch


def download_roberta(model):
    os.system(f"wget https://dl.fbaipublicfiles.com/fairseq/models/{model}.tar.gz")
    os.system(f"tar -xzvf {model}.tar.gz")


def download_gpt2(model):
    print("Saving gpt2-medium embeddings matrix...")
    if not os.path.exists(f"{model}"):
        os.makedirs(f"{model}")
    from transformers import AutoModelWithLMHead, GPT2Model

    gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2-medium")
    torch.save(gpt2_model.base_model.wte.weight,f'{model}/gpt2-medium-emb.pt')

def create_roberta_dict(mod_path):
    print("Creating RoBERTa real_dict.txt...")
    from fairseq.models.roberta import RobertaModel
    roberta = RobertaModel.from_pretrained(mod_path, checkpoint_file="model.pt")
    sorted_roberta_dict = []
    for idx, token in roberta.bpe.bpe.decoder.items():
        sorted_roberta_dict.append(token)

    real_dict = open(mod_path + 'real_dict.txt', 'w+')
    with open(mod_path + 'dict.txt', 'r') as dict_f:  # with open(dict_name, 'r', encoding='utf8') as input_f:
        for line in dict_f.readlines():
            # print(f"line:{line}")
            token = line.split(' ')[0]
            if 'madeupword' in token:
                continue
            token = sorted_roberta_dict[int(token)]
            real_dict.write(token)
            real_dict.write(' ' + str(line.split(' ')[1]))

    real_dict.close()


def sort_dict_by_value(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}


def write_dict_to_file(d, filename):
    with open(filename, 'w+') as f:
        for key, value in d.items():
            f.write(f'{key} {value}\n')


def create_gpt2_dict(mod_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    write_dict_to_file(sort_dict_by_value(tokenizer.vocab), mod_path + 'dict.txt')


def create_spelling_data(model, real_dict_path):
    if not os.path.exists("spelling_data"):
        os.makedirs("spelling_data")
    spelling_data_path = 'spelling_data/' + f'tokens2char.{model}.txt'

    print(f"Creating Spelling data at: {spelling_data_path} ...")

    spelling_data_file = open(spelling_data_path, 'w+')  # , encoding='utf8')

    with open(real_dict_path, 'r') as dict_f:
        for line in dict_f.readlines():
            token = line.split(' ')[0]
            spelling_data_file.write(token)
            for c in token:
                spelling_data_file.write(' ' + c)
            spelling_data_file.write(' \n')

    spelling_data_file.close()
    print("Done!")


def run_main(args):
    if args.model_dir is None:
        args.model_dir = os.getcwd()
    mod_path = os.path.join(args.model_dir, f"{args.model}/")

    if 'roberta' in args.model:
        # download_roberta(args.model)
        create_roberta_dict(mod_path)
        create_spelling_data(args.model, mod_path + 'real_dict.txt')
    if 'gpt2' in args.model:
        download_gpt2(args.model)
        create_gpt2_dict(mod_path)
        create_spelling_data(args.model, mod_path + 'dict.txt')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir", type=str, default=None, help=""
    )

    parser.add_argument(
        "--model",
        type=str,
        default="roberta.base",
        help="model can be roberta.base, roberta.large, gpt2, arabert or glove.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    run_main(args)
