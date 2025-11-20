#!/bin/bash
cd /work/nvme/bbjs/ttao3/espnet/egs2/titw/tts1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
/work/hdd/bbjs/ttao3/.conda/envs/espnet310/bin/python -m espnet2.bin.tts_train --use_preprocessor true --token_type phn --token_list dump/24k/token_list/phn_tacotron_g2p_en_no_space/tokens.txt --non_linguistic_symbols none --cleaner tacotron --g2p g2p_en_no_space --normalize global_mvn --resume true --fold_length 150 --fold_length 204800 --output_dir exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space --config conf/tuning/train_f5tts.yaml --feats_extract fbank --feats_extract_conf n_fft=1024 --feats_extract_conf hop_length=256 --feats_extract_conf win_length=1024 --feats_extract_conf fs=24000 --feats_extract_conf fmin=80 --feats_extract_conf fmax=7600 --feats_extract_conf n_mels=100 --train_data_path_and_name_and_type dump/24k/raw/titw_easy_ks_train/text,text,text --train_data_path_and_name_and_type dump/24k/raw/titw_easy_ks_train/wav.scp,speech,sound --train_shape_file exp_ks/tts_stats_raw_phn_tacotron_g2p_en_no_space/train/text_shape.phn --train_shape_file exp_ks/tts_stats_raw_phn_tacotron_g2p_en_no_space/train/speech_shape --valid_data_path_and_name_and_type dump/24k/raw/titw_easy_ks_dev/text,text,text --valid_data_path_and_name_and_type dump/24k/raw/titw_easy_ks_dev/wav.scp,speech,sound --valid_shape_file exp_ks/tts_stats_raw_phn_tacotron_g2p_en_no_space/valid/text_shape.phn --valid_shape_file exp_ks/tts_stats_raw_phn_tacotron_g2p_en_no_space/valid/speech_shape --normalize_conf stats_file=exp_ks/tts_stats_raw_phn_tacotron_g2p_en_no_space/train/feats_stats.npz --ngpu 2 --multiprocessing_distributed True 
EOF
) >exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/train.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/train.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( /work/hdd/bbjs/ttao3/.conda/envs/espnet310/bin/python -m espnet2.bin.tts_train --use_preprocessor true --token_type phn --token_list dump/24k/token_list/phn_tacotron_g2p_en_no_space/tokens.txt --non_linguistic_symbols none --cleaner tacotron --g2p g2p_en_no_space --normalize global_mvn --resume true --fold_length 150 --fold_length 204800 --output_dir exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space --config conf/tuning/train_f5tts.yaml --feats_extract fbank --feats_extract_conf n_fft=1024 --feats_extract_conf hop_length=256 --feats_extract_conf win_length=1024 --feats_extract_conf fs=24000 --feats_extract_conf fmin=80 --feats_extract_conf fmax=7600 --feats_extract_conf n_mels=100 --train_data_path_and_name_and_type dump/24k/raw/titw_easy_ks_train/text,text,text --train_data_path_and_name_and_type dump/24k/raw/titw_easy_ks_train/wav.scp,speech,sound --train_shape_file exp_ks/tts_stats_raw_phn_tacotron_g2p_en_no_space/train/text_shape.phn --train_shape_file exp_ks/tts_stats_raw_phn_tacotron_g2p_en_no_space/train/speech_shape --valid_data_path_and_name_and_type dump/24k/raw/titw_easy_ks_dev/text,text,text --valid_data_path_and_name_and_type dump/24k/raw/titw_easy_ks_dev/wav.scp,speech,sound --valid_shape_file exp_ks/tts_stats_raw_phn_tacotron_g2p_en_no_space/valid/text_shape.phn --valid_shape_file exp_ks/tts_stats_raw_phn_tacotron_g2p_en_no_space/valid/speech_shape --normalize_conf stats_file=exp_ks/tts_stats_raw_phn_tacotron_g2p_en_no_space/train/feats_stats.npz --ngpu 2 --multiprocessing_distributed True  ) &>>exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/train.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/train.log
echo '#' Accounting: end_time=$time2 >>exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/train.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/train.log
echo '#' Finished at `date` with status $ret >>exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/train.log
[ $ret -eq 137 ] && exit 100;
touch exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/q/done.2829386
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  --job-name exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/train.log --time 48:00:00 --account=bbjs-delta-gpu --partition=gpuA100x4 --nodes=1 --ntasks-per-node=1 --gres=gpu:a100:2 --cpus-per-task=16 --mem=30g --gpu-bind=closest --constraint=scratch --cpus-per-task 64 --mem=60800M  --open-mode=append -e exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/q/train.log -o exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/q/train.log  /work/nvme/bbjs/ttao3/espnet/egs2/titw/tts1/exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/q/train.sh >>exp_ks/tts_train_f5tts_raw_phn_tacotron_g2p_en_no_space/q/train.log 2>&1
