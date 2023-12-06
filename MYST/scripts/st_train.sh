PROJECT_ROOT="${HOME}"/project/MYST
fairseq-hydra-train \
task.data="${HOME}"/dataset/mustc \
task.config_yaml="${HOME}"/dataset/mustc/en-de/config_st.yaml \
tensorboard_logdir="${PROJECT_ROOT}"/st_train_log \
checkpoint.save_dir="${PROJECT_ROOT}"/checkpoint \
checkpoint.restore_file="${PROJECT_ROOT}"/checkpoint_last.pt \
model.w2v_model_path="${PROJECT_ROOT}"/checkpoints/wav2vec_small.pt \
--config-dir "${PROJECT_ROOT}"/MYST/config \
--config-name mutitask_en_de