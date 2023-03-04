import os
from invoke import run
import itertools

checkpoints_dir = "/home/olab/itayitzhak/bpeplus/fairseq/checkpoints/"
time_request = "1440"#"1440"#"40"#
partition = "killable" #"gpu-levyomer" #

def run_main():
    exp_name ="GloVe" #"roberta.base"  # "GloVe"#"gpt2"#"roberta.large"#"AraBERT" #"roberta.large" ##"AraBERT"  # "_wikitext_char_emb_knn_config"
    lang = "GloVe"#'character-bert'#"roberta.base"#  # "GloVe"#"gpt2.medium"#"roberta.base" #"roberta.large"#"AraBERT"##"roberta.base"
    data_lang = "GloVe"#'roberta.base'
    whole_words = ""#"_whole_words"  # ""
    train_or_eval = 'eval'

    load_emb_options = ["trained", "not_trained"]#["trained"]#["trained", "not_trained"]
    seeds = [0,1,2,3,4,5,6,7,8,9]
    splits = ['random_split', 'lemma_similarity_split']#['similarity_split']#['random_split', 'similarity_split', 'lemma_similarity_split']
    train_sizes = [32000]#[1000, 2000, 4000, 8000, 16000, 32000]
    #freq_list = ["_freq_range_0_10000_","_freq_range_10000_20000_", "_freq_range_20000_30000_", "_freq_range_30000_40000_","_freq_range_40000_50000_"]
    freq_list = [""]

    for seed, split_state, train_size, is_load_emb, freq_name in list(
            itertools.product(seeds, splits, train_sizes, load_emb_options, freq_list)):
        configuration = get_configuration(lang, data_lang, exp_name, train_or_eval, seed, split_state, train_size, is_load_emb, freq_name, whole_words)
        create_files_and_run(configuration)

##############################################################
def get_configuration(lang, data_lang, exp_name, train_or_eval, seed, split_state, train_size, is_load_emb, freq_name, whole_words):
    conf = dict()
    conf[
        'data_split_name'] = f'{seed}_{split_state}_train_size_{train_size}{freq_name}{whole_words}'  # "lemma_similarity_split_seed_0_0.5"
    conf['exp_name'] = exp_name
    conf['task'] = "LM"
    conf['train_or_eval'] = train_or_eval
    conf['lang'] = lang
    conf['data_lang'] = data_lang
    conf['add_char_emb'] = ""  # "--add-token-unicode-characters-embeddings"
    conf['is_load_emb'] = is_load_emb
    conf['data'] = ""  # "wiki"
    conf['data_folder'] = "spelling_data/organized_tests/"  # "wiki"
    conf['checkpoint_type'] = "checkpoint_last" # ""checkpoint_best"


    if conf['task'] == "translation":
        conf['bpe30k'] = ""  # ".bpe30k"
        conf['save_dir'] = conf['lang'] + conf['bpe30k'] + conf['data_split_name']  # have not done learned
        conf['data_name'] = "iwslt14.tokenized." + conf['lang'] + conf['bpe30k']
    elif conf['task'] == "LM":
        conf['warm_ups'] = 100  # scan1000
        conf['max_updates'] = 10000  # 10000
        conf['save_interval'] = "--save-interval 500"
        conf['with_shtrudel'] = ""  # ".with@"
        conf['reset_chars_emb'] = ""  # " --reset-chars-emb" # ""
        conf['model_size'] = ""  # "   --decoder-layers 12" #""
        conf['decoder_embed_dim'] = "512"
        conf['--no-epoch-checkpoints'] = "--no-epoch-checkpoints"
        conf['--disable-validation'] = '--disable-validation'

    if conf['is_load_emb'] == "trained":
        conf[
            'load_embeddings'] = f"--path-to-load-embeddings /home/olab/itayitzhak/bpeplus/fairseq/checkpoints/{conf['lang']}-en_baseline/checkpoint_last.pt "
        conf['fp16'] = " --fp16"
    elif conf['is_load_emb'] == "not_trained":
        conf['load_embeddings'] = ""
        conf['fp16'] = ""

    if "roberta" in conf['lang'] or "gpt2.medium" == conf['lang'] or "AraBERT" == conf['lang'] or "GloVe" == conf['lang'] or 'character-bert' == conf['lang']:
        if conf['exp_name'] == "roberta.large" or conf['exp_name'] == "gpt2" or conf['exp_name'] == "AraBERT":
            conf[
                'emb_dim'] = f" --decoder-output-dim 1024  --decoder-input-dim 1024 --decoder-embed-dim {conf['decoder_embed_dim']} {conf['fp16']}"
        elif "GloVe" == conf['lang']:
            conf[
                'emb_dim'] = f" --decoder-output-dim 300  --decoder-input-dim 300 --decoder-embed-dim {conf['decoder_embed_dim']} {conf['fp16']}"
        else: # roberta.base,character-bert
            conf[
                'emb_dim'] = f" --decoder-output-dim 768  --decoder-input-dim 768 --decoder-embed-dim {conf['decoder_embed_dim']} {conf['fp16']}"
    else:
        conf['emb_dim'] = ""

    if conf['data'] == "wiki":
        conf['data_name'] = "wikitext-103"
        conf['sample_break_mode'] = "none"
        conf['save_dir'] = conf['is_load_emb'] + conf['exp_name'] + "_wiki_" + conf['add_char_emb'].strip().replace(" ",
                                                                                                                    "_")
    else:
        conf[
            'data_name'] = f"{conf['data_folder']}tokens2char.{conf['data_lang']}{conf['with_shtrudel']}.txt.{conf['data_split_name']}"
        conf['sample_break_mode'] = "eos"
        conf['save_dir'] = conf['is_load_emb'] + '_' + conf['lang'] + '_' + conf['data_split_name'] + conf[
            'with_shtrudel'] + conf['reset_chars_emb'].strip() + conf['model_size'].strip().replace(
            " ", "_") + conf['exp_name'] + str(f"_decoder_embed_dim_{conf['decoder_embed_dim']}")

    return conf


def get_run_file_text(script_file, train_or_eval, save_dir, data_name):
    run_file_text = "#!/bin/sh\n" + \
                    "#SBATCH --job-name=" + save_dir + "\n" + \
                    "#SBATCH --output=" + checkpoints_dir + save_dir + "/" + train_or_eval + ".out\n" + \
                    "#SBATCH --error=" + checkpoints_dir + save_dir + "/" + train_or_eval + ".err\n" + \
                    f"#SBATCH --partition={partition}\n" + \
                    "#SBATCH --signal=USR1@120\n" + \
                    "#SBATCH --mem=10000\n" + \
                    f"#SBATCH --time={time_request}\n" + \
                    "#SBATCH --ntasks=1\n" + \
                    "#SBATCH --nodes=1\n" + \
                    "#SBATCH --gpus-per-task=1\n" + \
                    "#SBATCH --cpus-per-task=2\n" + \
                    "srun sh " + script_file + "\n"
    return run_file_text


def translation_get_script_file_text(train_or_eval, save_dir, data_name, add_char_emb):
    if train_or_eval is "train":
        script_file_text = "#!/usr/bin/env bash\n\n" + \
                           "/home/olab/itayitzhak/anaconda3/envs/bpeplus4/bin/python /home/olab/itayitzhak/bpeplus/fairseq/train.py \\\n" + \
                           "  /home/olab/itayitzhak/bpeplus/fairseq/data-bin/" + data_name + " \\\n" + \
                           "  --arch transformer_iwslt_de_en \\\n" + \
                           "  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \\\n" + \
                           "  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \\\n" + \
                           "  --dropout 0.3 --weight-decay 0.0001 \\\n" + \
                           "  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \\\n" + \
                           "  --max-tokens 4096 \\\n" + \
                           "  --max-update 60000 \\\n" + \
                           "  --eval-bleu \\\n" + \
                           "  --eval-bleu-args \'{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}\' \\\n" + \
                           "  --eval-bleu-detok moses \\\n" + \
                           "  --eval-bleu-remove-bpe \\\n" + \
                           "  --share-all-embeddings --no-epoch-checkpoints  \\\n" + \
                           "  --eval-bleu-remove-bpe \\\n" + \
                           "  --eval-bleu-remove-bpe \\\n" + \
                           "  " + add_char_emb + " \\\n" + \
                           "  --save-dir " + checkpoints_dir + save_dir

    elif train_or_eval is "eval":
        script_file_text = "#!/usr/bin/env bash\n" + \
                           "  /home/olab/itayitzhak/anaconda3/envs/bpeplus4/bin/python /home/olab/itayitzhak/bpeplus/fairseq/fairseq_cli/generate.py \\\n" + \
                           "  /home/olab/itayitzhak/bpeplus/fairseq/data-bin/" + data_name + " \\\n" + \
                           "  --path " + checkpoints_dir + save_dir + f"/checkpoint_best.pt \\\n" + \
                           "  --batch-size 128 --beam 5 --remove-bpe"

    return script_file_text


def LM_get_script_file_text(train_or_eval, conf):
    if train_or_eval is "train":
        if conf['data'] is not "wiki":
            script_file_text = "#!/usr/bin/env bash\n\n" + \
                               "/home/olab/itayitzhak/anaconda3/envs/bpeplus4/bin/python /home/olab/itayitzhak/bpeplus/fairseq/train.py \\\n" + \
                               "  --task language_modeling" + " \\\n" + \
                               "  /home/olab/itayitzhak/bpeplus/fairseq/data-bin/" + conf['data_name'] + " \\\n" + \
                               "  --arch transformer_lm \\\n" + \
                               "  --optimizer adam --clip-norm 0.0 \\\n" + \
                               f"  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates {conf['warm_ups']} \\\n" + \
                               "  --dropout 0.1 --weight-decay 0.01 \\\n" + \
                               f"  --warmup-init-lr 1e-07 --tokens-per-sample 512 --sample-break-mode {conf['sample_break_mode']}  --max-tokens 1024 --update-freq 16\\\n" + \
                               "  --share-decoder-input-output-embed \\\n" + \
                               f" --max-update {conf['max_updates']} {conf['emb_dim']} {conf['model_size']} {conf['--no-epoch-checkpoints']} {conf['--disable-validation']}\\\n" + \
                               f" {conf['load_embeddings']} {conf['reset_chars_emb']} {conf['add_char_emb']}\\\n" + \
                               f"  {conf['save_interval']} \\\n" + \
                               " --dont-send-comet-ml --save-dir " + checkpoints_dir + conf['save_dir']
        else:
            script_file_text = "#!/usr/bin/env bash\n\n" + \
                               "/home/olab/itayitzhak/anaconda3/envs/bpeplus4/bin/python /home/olab/itayitzhak/bpeplus/fairseq/train.py \\\n" + \
                               "  --task language_modeling" + " \\\n" + \
                               "  /home/olab/itayitzhak/bpeplus/fairseq/data-bin/" + conf['data_name'] + " \\\n" + \
                               "  --arch transformer_lm_wiki103 \\\n" + \
                               "  --optimizer nag --clip-norm 0.1 --t-mult 2 \\\n" + \
                               f" --lr-period-updates 270000 --lr-shrink 0.75 --lr 1e-4 --lr-scheduler cosine --warmup-updates 16000 \\\n" + \
                               "  --criterion adaptive_loss \\\n" + \
                               f"  --warmup-init-lr 1e-07 --min-lr 1e-09 --tokens-per-sample 512 --sample-break-mode none  --max-tokens 600 --update-freq 3 \\\n" + \
                               "  --share-decoder-input-output-embed --seed 1 \\\n" + \
                               f" --max-update 286000 --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \\\n" + \
                               "  --save-dir " + checkpoints_dir + conf['save_dir']

    elif train_or_eval is "eval":
        script_file_text = "#!/usr/bin/env bash\n" + \
                           "  /home/olab/itayitzhak/anaconda3/envs/bpeplus4/bin/python /home/olab/itayitzhak/bpeplus/fairseq/fairseq_cli/generate.py \\\n" + \
                           "  /home/olab/itayitzhak/bpeplus/fairseq/data-bin/" + conf['data_name'] + " \\\n" + \
                           "  --task language_modeling" + " \\\n" + \
                           "  --path " + checkpoints_dir + conf['save_dir'] + f"/{conf['checkpoint_type']}.pt \\\n" + \
                           "  --skip-invalid-size-inputs-valid-test --prefix-size 1" + " \\\n" + \
                           f"  --batch-size 10 --beam 1 --sample-break-mode eos {conf['fp16']}" + " \\\n" + \
                           "  --results-path " + checkpoints_dir + conf['save_dir']

    return script_file_text


def create_files_and_run(conf):
    save_dir = conf['save_dir']
    train_or_eval = conf['train_or_eval']
    data_name = conf['train_or_eval']
    task = conf['task']
    add_char_emb = conf['add_char_emb']

    try:
        os.mkdir(checkpoints_dir + save_dir)
    except OSError as error:
        print(error)

    run_file = "/specific/a/home/cc/cs/itayitzhak/Slurm/run_" + save_dir + conf['exp_name']
    script_file = "/specific/a/home/cc/cs/itayitzhak/Slurm/" + save_dir + "_command_" + train_or_eval + "_" + task + \
                  conf['exp_name'] + "_bpeplus.sh"
    run_file_text = get_run_file_text(script_file, train_or_eval, save_dir, data_name)

    if task == "translation":
        script_file_text = translation_get_script_file_text(train_or_eval, save_dir, data_name, add_char_emb)
    if task == "LM":
        script_file_text = LM_get_script_file_text(train_or_eval, conf)

    with open(run_file, 'w+') as f:
        f.write(run_file_text)

    with open(script_file, 'w+') as f:
        f.write(script_file_text)

    run("sbatch " + run_file)
    print("=====" + task + " Task======")
    print("run_file:", run_file)
    print("script_file:", script_file)
    print("Using data:", data_name)
    print(train_or_eval)


if __name__ == "__main__":
    run_main()
