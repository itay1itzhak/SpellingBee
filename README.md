This repository is the code base for the paper: [Models In a Spelling Bee: Language Models Implicitly Learn the Character Composition of Tokens (NAACL 2022)](https://arxiv.org/abs/2108.11193) and is based on the [fairseq](https://github.com/pytorch/fairseq) code base. 

## Requirements

* [PyTorch](http://pytorch.org/) version >= 1.9.0
* Python version >= 3.7

View requirments.txt file for more information.

## Getting started
```bash
git clone https://github.com/itay1itzhak/SpellingBee.git
cd SpellingBee
pip install --editable ./
pip install transformers 
pip install nltk 
python SpellingBee\nltk_package_downloader.py
pip install farasapy # for probing the AraBERT model
```

## Download embeddings and prepare data
Use the following scripts to download the model embeddings and dictionary and prepare data for the SpellingBee probe.
MODEL_NAME can be one of following - roberta.base, roberta.large, or gpt2. Additional models (AraBERT, GloVe and CharacterBERT) will be added in the future.
```bash
cd SpellingBee
python download_pretrained_embeddings.py --model $MODEL_NAME 
python create_spelling_data.py --model $MODEL_NAME
```
## SpelllingBee probe
Now you can use the default fairseq commands to preprocess the spelling data, train and evaluate the SpellingBee probes.

Here are examples for train and evaluate commands for one seed in a specific seed, split (Filter) and train size. In order the get the full results a probe should be trained and evaluate for every seed split and train size wanted.

## Preprocess

You can edit and use the preprocess.sh script to run preprocess according to the spelling data you created.
[EMB_DIM_SIZE should be 1024 for robert.large, gpt2, arabert, and 300 for glove.]
```bash
 cd ..
 MODEL_NAME=roberta.base \
 DICT_PATH=SpellingBee/roberta.base/real_dict.txt \
 EMB_DIM_SIZE=768 \
 SEED=1 \
 SPLIT=random_split \ 
 TRAIN_SIZE=32000 
 
fairseq-preprocess \
 --only-source \
 --trainpref SpellingBee/spelling_data/tokens2char.$MODEL_NAME.txt.1\_$SPLIT\_train\_size\_$TRAIN_SIZE.train \
  --validpref SpellingBee/spelling_data/tokens2char.$MODEL_NAME.txt.1\_$SPLIT\_train\_size\_$TRAIN_SIZE.valid \
 --testpref SpellingBee/spelling_data/tokens2char.$MODEL_NAME.txt.1\_$SPLIT\_train\_size\_$TRAIN_SIZE.test \
 --destdir SpellingBee/spelling_data/tokens2char.$MODEL_NAME.txt.1\_$SPLIT\_train\_size\_$TRAIN_SIZE \
 --workers 20 \
 --tokenizer space \
 --srcdict $DICT_PATH
```
## Train and evaluate

```bash
 MODEL_TO_LOAD=roberta.base \
 MODEL_TO_LOAD_PATH=SpellingBee/roberta.base/model.pt \
 python train.py \
  --task language_modeling \
  SpellingBee/spelling_data/tokens2char.$MODEL_NAME.txt.1\_$SPLIT\_train\_size\_$TRAIN_SIZE \
  --arch transformer_lm \
  --optimizer adam --clip-norm 0.0 \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 100 \
  --dropout 0.1 --weight-decay 0.01 \
  --warmup-init-lr 1e-07 --tokens-per-sample 512 --sample-break-mode eos  --max-tokens 1024 --update-freq 16 --fp16 \
  --share-decoder-input-output-embed \
  --max-update 10000  --decoder-output-dim $EMB_DIM_SIZE  --decoder-input-dim $EMB_DIM_SIZE --decoder-embed-dim 512 --no-epoch-checkpoints --disable-validation\
  --model-to-load-embeddings $MODEL\_TO\_LOAD \
  --path-to-load-embeddings  $MODEL\_TO\_LOAD_PATH  \
  --save-dir checkpoints/pretrained_embeddings_$MODEL_NAME.txt.1\_$SPLIT\_train\_size\_$TRAIN_SIZE
```
To train a control test, a probe can be trained without loading pretrained embeddings by removing the arguments model-to-load-embeddings and path-to-load-embeddings.

[Note - the prefix for a probe trained with pretrained embeddings and control must the same as you enter in results analysis with the arguments --trained_emb_prefix (default is 'pretrained') and --random_init_emb_prefix (default is None and not considered).]

To generate evaluation of the spelling of the probe use the fairseq-generate command. 
```bash
  python fairseq_cli/generate.py \
  SpellingBee/spelling_data/tokens2char.$MODEL_NAME.txt.1\_$SPLIT\_train\_size\_$TRAIN_SIZE \
  --task language_modeling \
  --path checkpoints/pretrained_embeddings_$MODEL_NAME.txt.1\_$SPLIT\_train\_size\_$TRAIN_SIZE/checkpoint\_last.pt \
  --skip-invalid-size-inputs-valid-test --prefix-size 1 \
  --batch-size 10 --beam 1 --sample-break-mode eos  --fp16 \
  --results-path checkpoints/pretrained_embeddings_$MODEL_NAME.txt.1\_$SPLIT\_train\_size\_$TRAIN_SIZE
```

## Results analysis
Run the analysis script to create a csv file with the results across random seeds. The results will be saved in SpellingBee/results.
```bash
cd SpellingBee
python results_analysis.py --model $MODEL_NAME
```
## Training embeddings to spell for language model pretraining
The SpellingBee probe can be used to pretrain embeddings to spell before training a language model.
This can be done creating training spelling data with ALL tokens in the dictionary, training a SpellingBee probe on it, and then start a language model training with the pretrained embeddings. An example for training a RoBERTa model with pretrained embeddings is described below.

Creating spelling data with all tokens:
```bash
  fairseq-preprocess \
  --only-source  \
 --trainpref SpellingBee/spelling_data/tokens2char.$MODEL_NAME.txt \
 --validpref SpellingBee/spelling_data/tokens2char.$MODEL_NAME.txt  \
 --testpref SpellingBee/spelling_data/tokens2char.$MODEL_NAME.txt \
 --destdir SpellingBee/spelling_data/tokens2char.$MODEL_NAME \
 --workers 20    \
 --tokenizer space \
 --srcdict $DICT_PATH 
```
Training
```bash
 MODEL_TO_LOAD=roberta.base \
 MODEL_TO_LOAD_PATH=SpellingBee/roberta.base/model.pt \
 python train.py \
  --task language_modeling \
  SpellingBee/spelling_data/tokens2char.$MODEL_NAME \
  --arch transformer_lm \
  --optimizer adam --clip-norm 0.0 \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 100 \
  --dropout 0.1 --weight-decay 0.01 \
  --warmup-init-lr 1e-07 --tokens-per-sample 512 --sample-break-mode eos  --max-tokens 1024 --update-freq 16 --fp16 \
  --share-decoder-input-output-embed \
  --max-update 10000  --decoder-output-dim $EMB_DIM_SIZE  --decoder-input-dim $EMB_DIM_SIZE --decoder-embed-dim 512 --no-epoch-checkpoints --disable-validation\
  --save-dir checkpoints/pretrained_by_spellingBee_$MODEL_NAME
```
Verifying the embeddings contain enough spelling information using a SpellingBee probe with a different seed
```bash
 SEED=43 \
  python train.py \
  --task language_modeling \
  SpellingBee/spelling_data/tokens2char.$MODEL_NAME \
  --arch transformer_lm \
  --optimizer adam --clip-norm 0.0 \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 100 \
  --dropout 0.1 --weight-decay 0.01 \
  --warmup-init-lr 1e-07 --tokens-per-sample 512 --sample-break-mode eos  --max-tokens 1024 --update-freq 16 --fp16 \
  --share-decoder-input-output-embed \
  --seed $SEED \
  --model-to-load-embeddings spellingbee_pretrained \
  --path-to-load-embeddings checkpoints/pretrained_by_spellingBee_$MODEL_NAME/checkpoint_last.pt   \
  --max-update 10000  --decoder-output-dim 1024  --decoder-input-dim 1024 --decoder-embed-dim 512 --no-epoch-checkpoints --disable-validation\
  --save-dir checkpoints/verify_pretrained_by_spellingBee_$MODEL_NAME
```

```bash
 python fairseq_cli/generate.py \
 SpellingBee/spelling_data/tokens2char.$MODEL_NAME \
--task language_modeling \
--path checkpoints/verify_pretrained_by_spellingBee_$MODEL_NAME/checkpoint_last.pt \
--skip-invalid-size-inputs-valid-test --prefix-size 1 \
--batch-size 10 --beam 1 --sample-break-mode eos --fp16 \
--results-path checkpoints/verify_pretrained_by_spellingBee_$MODEL_NAME
--results-path checkpoints/verify_pretrained_by_spellingBee_$MODEL_NAME
```

And finally training a RoBERTa-like language model with the pretrained embeddings
```bash
TOTAL_UPDATES=50000    # Total number of training steps
WARMUP_UPDATES=5000    # Warmup the learning rate over this many updates
PEAK_LR=2e-3          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=128   # Max sequence length
MAX_POSITIONS=128       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=64        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x
DATA_DIR=path/to/data-for-pretraining \
python train.py --fp16 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_large --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
    --validate-interval-updates 500 --skip-invalid-size-inputs-valid-test  --encoder-normalize-before \
    --model-to-load-embeddings spellingbee_pretrained \
    --path-to-load-embeddings checkpoints/pretrained_by_spellingBee_$MODEL_NAME/checkpoint_last.pt \
    --save-dir checkpoints/$MODEL_NAME_pre_training
```
## License

SpelllingBee is MIT-licensed.
