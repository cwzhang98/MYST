fairseq-train ~/dataset/wmt16/en-de/bin \
--arch transformer_wmt_en_de --share-decoder-input-output-embed --share-all-embeddings \
--encoder-normalize-before --decoder-normalize-before \
--task translation --source-lang en --target-lang de \
--fp16 --ddp-backend no_c10d --distributed-world-size 2 --update-freq 1 \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 10.0 \
--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 25000 \
--dropout 0.1 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --ignore-prefix-size 1 \
--max-tokens 40000 --eval-bleu --eval-bleu-args '{"beam": 5, "prefix_size": 1}' \
--eval-bleu-bpe --spm-path /home2/cwzhang98/dataset/mustc/en-de/spm_unigram_10000_st.model \
--eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--save-dir checkpoint_mt --restore-file checkpoint_mt/checkpoint_last.pt \
--log-interval 100 --patience 10 --tensorboard-logdir mt_train_log --log-format json \
--keep-last-epochs 15