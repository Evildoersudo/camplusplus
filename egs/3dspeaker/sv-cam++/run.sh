#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
. ./path.sh || exit 1

# Resolve the recipe directory explicitly so Python imports do not depend on
# how the shell normalizes relative paths on Windows/Git Bash.
recipe_dir=$(cd "$(dirname "$0")" && pwd)
project_root=$(cd "${recipe_dir}/../../.." && pwd)

# Control which stages of the recipe to execute.
stage=1
stop_stage=5

# Common experiment paths and GPU selection.
data=data
exp=exp
exp_name=cam++
gpus="0 1 2 3"
dataset=3dspeaker

# VCTK-specific options. These are ignored when dataset=3dspeaker.
vctk_resampled_dir=data/vctk_16k/wav16k
vctk_max_speakers=0
vctk_max_utts_per_speaker=0
vctk_num_train_utts=250
vctk_num_test_utts=50
vctk_num_target_trials=20
vctk_num_nontarget_trials=20
vctk_seed=1234

. utils/parse_options.sh || exit 1

exp_dir=$exp/$exp_name
num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
speakerlab_root=${project_root}/speakerlab
export PYTHONPATH=${project_root}:${PYTHONPATH}
if [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/python.exe" ]; then
  # Prefer the currently activated Conda environment on Windows.
  python_cmd="${CONDA_PREFIX}/python.exe"
else
  python_cmd="python"
fi

if [ "${dataset}" = "3dspeaker" ] && [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Prepare 3D-Speaker raw data and Kaldi-style metadata.
  echo "Stage1: Preparing 3D Speaker dataset..."
  ./local/prepare_data.sh --stage 1 --stop_stage 3 --data ${data} --dataset ${dataset}
fi

if [ "${dataset}" = "vctk" ] && [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Build VCTK train/test splits and generate trials in one call.
  echo "Stage1: Preparing VCTK train/test split..."
  ./local/prepare_data.sh \
    --dataset ${dataset} \
    --stage 3 \
    --stop_stage 4 \
    --data ${data} \
    --vctk_resampled_dir ${vctk_resampled_dir} \
    --vctk_max_speakers ${vctk_max_speakers} \
    --vctk_max_utts_per_speaker ${vctk_max_utts_per_speaker} \
    --vctk_num_train_utts ${vctk_num_train_utts} \
    --vctk_num_test_utts ${vctk_num_test_utts} \
    --vctk_num_target_trials ${vctk_num_target_trials} \
    --vctk_num_nontarget_trials ${vctk_num_nontarget_trials} \
    --vctk_seed ${vctk_seed}
fi

if [ "${dataset}" = "3dspeaker" ] && [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Convert Kaldi metadata into the CSV format used by the training dataset.
  echo "Stage2: Preparing training data index files..."
  ${python_cmd} local/prepare_data_csv.py --data_dir $data/3dspeaker/train
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Train CAM++ from the dataset-specific training CSV.
  echo "Stage3: Training the speaker model..."
  if [ "${dataset}" = "3dspeaker" ]; then
    train_csv=$data/3dspeaker/train/train.csv
    noise_scp=$data/musan/wav.scp
    reverb_scp=$data/rirs/wav.scp
    aug_prob=0.8
  elif [ "${dataset}" = "vctk" ]; then
    train_csv=$data/vctk/train/train.csv
    noise_scp=
    reverb_scp=
    # VCTK runs without external MUSAN/RIRS augmentation by default.
    aug_prob=0.0
  else
    echo "Unsupported dataset: ${dataset}"
    exit 1
  fi

  if [ ${num_gpu} -le 1 ]; then
    # Single-GPU mode avoids torchrun/DDP, which is more reliable on Windows.
    (
      cd ${project_root}
      ${python_cmd} -m speakerlab.bin.train --config ${recipe_dir}/conf/cam++.yaml --gpu $gpus \
             --data ${recipe_dir}/$train_csv --noise "$noise_scp" --reverb "$reverb_scp" --aug_prob $aug_prob --exp_dir ${recipe_dir}/$exp_dir
    )
  else
    (
      cd ${project_root}
      torchrun --nproc_per_node=$num_gpu -m speakerlab.bin.train --config ${recipe_dir}/conf/cam++.yaml --gpu $gpus \
               --data ${recipe_dir}/$train_csv --noise "$noise_scp" --reverb "$reverb_scp" --aug_prob $aug_prob --exp_dir ${recipe_dir}/$exp_dir
    )
  fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Extract embeddings for every utterance in the dataset-specific test split.
  echo "Stage4: Extracting speaker embeddings..."
  if [ "${dataset}" = "3dspeaker" ]; then
    test_wav_scp=$data/3dspeaker/test/wav.scp
  elif [ "${dataset}" = "vctk" ]; then
    test_wav_scp=$data/vctk/test/wav.scp
  else
    echo "Unsupported dataset: ${dataset}"
    exit 1
  fi
  if [ ${num_gpu} -le 1 ]; then
    (
      cd ${project_root}
      ${python_cmd} -m speakerlab.bin.extract --exp_dir ${recipe_dir}/$exp_dir \
             --data ${recipe_dir}/$test_wav_scp --use_gpu --gpu $gpus
    )
  else
    (
      cd ${project_root}
      torchrun --nproc_per_node=$num_gpu -m speakerlab.bin.extract --exp_dir ${recipe_dir}/$exp_dir \
               --data ${recipe_dir}/$test_wav_scp --use_gpu --gpu $gpus
    )
  fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Score the extracted embeddings against the configured trial file(s).
  echo "Stage5: Computing score metrics..."
  if [ "${dataset}" = "3dspeaker" ]; then
    trials="$data/3dspeaker/trials/trials_cross_device $data/3dspeaker/trials/trials_cross_distance $data/3dspeaker/trials/trials_cross_dialect"
  elif [ "${dataset}" = "vctk" ]; then
    trials="$data/vctk/trials/trials"
  else
    echo "Unsupported dataset: ${dataset}"
    exit 1
  fi
  (
    cd ${project_root}
    ${python_cmd} -m speakerlab.bin.compute_score_metrics --enrol_data ${recipe_dir}/$exp_dir/embeddings --test_data ${recipe_dir}/$exp_dir/embeddings \
                                                   --scores_dir ${recipe_dir}/$exp_dir/scores --trials ${recipe_dir}/$trials
  )
fi
