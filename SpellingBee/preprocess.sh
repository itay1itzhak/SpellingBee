#!/bin/bash
MODEL="roberta.base"
SEEDS="1 2"
SPLITS="random_split similarity_split lemma_similarity_split"
TRAIN_SIZES="32000"
DICT_PATH="roberta.base/real_dict.txt"

echo "[${dt}] Preprocess for model ${MODEL}..."

for SEED in $SEEDS; do
    for SPLIT in $SPLITS; do
        for TRAIN_SIZE in $TRAIN_SIZES; do
              echo "[${dt}] Preprocess  on seed ${SEED} split ${SPLIT} train size ${TRAIN_SIZE}..."
              dt=$(date '+%d/%m/%Y %H:%M:%S')
              fairseq-preprocess \
              --only-source \
              --trainpref spelling_data/tokens2char.${MODEL}.txt.${SEED}_${SPLIT}_train_size_${TRAIN_SIZE}.train \
              --validpref spelling_data/tokens2char.${MODEL}.txt.${SEED}_${SPLIT}_train_size_${TRAIN_SIZE}.valid \
              --testpref spelling_data/tokens2char.${MODEL}.txt.${SEED}_${SPLIT}_train_size_${TRAIN_SIZE}.test \
              --destdir spelling_data/tokens2char.${MODEL}.txt.${SEED}_${SPLIT}_train_size_${TRAIN_SIZE} \
              --workers 20 \
              --tokenizer space \
              --srcdict ${DICT_PATH}
              dt=$(date '+%d/%m/%Y %H:%M:%S')
              #echo "[${dt}] Finished ${TEMPLATES}"
              #done
        done
    done
done