fairseq-hydra-train \
task.data=/home2/cwzhang98/project/dataset/LibriSpeech \
model.w2v_path=/home2/cwzhang98/project/MYST/checkpoints/wav2vec_small.pt \
--config-dir /home2/cwzhang98/project/MYST/examples/wav2vec/config/finetuning \
--config-name base_100h