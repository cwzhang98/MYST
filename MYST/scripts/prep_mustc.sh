#!/usr/bin/env bash

DATA_ROOT=$1
#MT_DATA_ROOT=$2
PROJECT_ROOT=$2
TGT_LANG=$3


TEXT_ROOT=${DATA_ROOT}/en-${TGT_LANG}/data/train/txt/train
MT_TEXT_ROOT=${MT_DATA_ROOT}/en-${TGT_LANG}/train

cd "$PROJECT_ROOT" || exit
# echo "converting mustc source text into phoneme"
# python examples/speech_text_joint_to_text/scripts/g2p_encode.py \
#    --lower-case --do-filter --use-word-start --no-punc \
#    --reserve-word examples/speech_text_joint_to_text/configs/mustc_noise.list \
#    --data-path "${TEXT_ROOT}".en \
#    --out-path "${TEXT_ROOT}"_pho.en

# echo "converting mt source text into phoneme"
# python examples/speech_text_joint_to_text/scripts/g2p_encode.py \
#     --lower-case --do-filter --use-word-start --no-punc\
#     --data-path "${MT_TEXT_ROOT}".en \
#     --out-path "${MT_TEXT_ROOT}"_pho.en

echo "preparing mustc dataset"
python examples/speech_to_text/prep_mustc_data.py \
   --data-root "${DATA_ROOT}" --lang de \
   --vocab-type unigram --vocab-size 8000