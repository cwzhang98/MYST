fairseq-hydra-train \
task.data=/home2/cwzhang98/dataset/LibriSpeech \
model.w2v_path=/home2/cwzhang98/project/MYST/checkpoints/wav2vec_small.pt \
--config-dir /home2/cwzhang98/project/MYST/MYST/config \
--config-name w2v_cif_pt.yaml