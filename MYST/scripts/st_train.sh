PROJECT_ROOT="${HOME}"/project/MYST
fairseq-hydra-train \
task.data="${HOME}"/dataset/mustc \
task.config_yaml="${HOME}"/dataset/mustc/en-de/config_st.yaml \
common.tensorboard_logdir="${PROJECT_ROOT}"/st_train_log \
checkpoint.save_dir="${PROJECT_ROOT}"/checkpoint \
checkpoint.restore_file="${PROJECT_ROOT}"/checkpoint/checkpoint_last.pt \
model.w2v_model_path="${PROJECT_ROOT}"/checkpoints/wav2vec_cif_10000.pt \
--config-dir "${PROJECT_ROOT}"/MYST/config \
--config-name mutitask_en_de