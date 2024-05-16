PROJECT_ROOT="${HOME}"/project/MYST
export CUDA_VISIBLE_DEVICES=2,3
fairseq-hydra-train \
task.data="${HOME}"/dataset/mustc \
common.tensorboard_logdir="${PROJECT_ROOT}"/st_train_log_es \
checkpoint.save_dir="${PROJECT_ROOT}"/checkpoint_es \
checkpoint.restore_file="${PROJECT_ROOT}"/checkpoint_es/checkpoint_last.pt \
model.w2v_model_path="${PROJECT_ROOT}"/checkpoints/wav2vec_small.pt \
criterion.consistency_weight=1.0 criterion.jsd_weight=1.0 \
--config-dir "${PROJECT_ROOT}"/MYST/config \
--config-name dpst_en_es