#!/bin/bash
cd /work/nvme/bbjs/ttao3/espnet/egs2/titw/tts1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
/work/hdd/bbjs/ttao3/.conda/envs/espnet310/bin/python -m espnet2.bin.tts_train --collect_stats true --write_collected_feats false --use_preprocessor true --token_type phn --token_list dump/24k/token_list/phn_tacotron_g2p_en_no_space/tokens.txt --non_linguistic_symbols none --cleaner tacotron --g2p g2p_en_no_space --normalize none --pitch_normalize none --energy_normalize none --train_data_path_and_name_and_type dump/24k/raw/titw_easy_train/text,text,text --train_data_path_and_name_and_type dump/24k/raw/titw_easy_train/wav.scp,speech,sound --valid_data_path_and_name_and_type dump/24k/raw/titw_easy_dev/text,text,text --valid_data_path_and_name_and_type dump/24k/raw/titw_easy_dev/wav.scp,speech,sound --train_shape_file exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/train.${SLURM_ARRAY_TASK_ID}.scp --valid_shape_file exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/valid.${SLURM_ARRAY_TASK_ID}.scp --output_dir exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/stats.${SLURM_ARRAY_TASK_ID} --config ./conf/tuning/train_f5tts.yaml --feats_extract fbank --feats_extract_conf n_fft=1024 --feats_extract_conf hop_length=240 --feats_extract_conf win_length=1024 --feats_extract_conf fs=24000 --feats_extract_conf fmin=0 --feats_extract_conf fmax=null --feats_extract_conf n_mels=100 --pitch_extract_conf fs=24000 --pitch_extract_conf n_fft=1024 --pitch_extract_conf hop_length=240 --pitch_extract_conf f0max=400 --pitch_extract_conf f0min=80 --energy_extract_conf fs=24000 --energy_extract_conf n_fft=1024 --energy_extract_conf hop_length=240 --energy_extract_conf win_length=1024 
EOF
) >exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/stats.$SLURM_ARRAY_TASK_ID.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/stats.$SLURM_ARRAY_TASK_ID.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( /work/hdd/bbjs/ttao3/.conda/envs/espnet310/bin/python -m espnet2.bin.tts_train --collect_stats true --write_collected_feats false --use_preprocessor true --token_type phn --token_list dump/24k/token_list/phn_tacotron_g2p_en_no_space/tokens.txt --non_linguistic_symbols none --cleaner tacotron --g2p g2p_en_no_space --normalize none --pitch_normalize none --energy_normalize none --train_data_path_and_name_and_type dump/24k/raw/titw_easy_train/text,text,text --train_data_path_and_name_and_type dump/24k/raw/titw_easy_train/wav.scp,speech,sound --valid_data_path_and_name_and_type dump/24k/raw/titw_easy_dev/text,text,text --valid_data_path_and_name_and_type dump/24k/raw/titw_easy_dev/wav.scp,speech,sound --train_shape_file exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/train.${SLURM_ARRAY_TASK_ID}.scp --valid_shape_file exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/valid.${SLURM_ARRAY_TASK_ID}.scp --output_dir exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/stats.${SLURM_ARRAY_TASK_ID} --config ./conf/tuning/train_f5tts.yaml --feats_extract fbank --feats_extract_conf n_fft=1024 --feats_extract_conf hop_length=240 --feats_extract_conf win_length=1024 --feats_extract_conf fs=24000 --feats_extract_conf fmin=0 --feats_extract_conf fmax=null --feats_extract_conf n_mels=100 --pitch_extract_conf fs=24000 --pitch_extract_conf n_fft=1024 --pitch_extract_conf hop_length=240 --pitch_extract_conf f0max=400 --pitch_extract_conf f0min=80 --energy_extract_conf fs=24000 --energy_extract_conf n_fft=1024 --energy_extract_conf hop_length=240 --energy_extract_conf win_length=1024  ) &>>exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/stats.$SLURM_ARRAY_TASK_ID.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/stats.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: end_time=$time2 >>exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/stats.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/stats.$SLURM_ARRAY_TASK_ID.log
echo '#' Finished at `date` with status $ret >>exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/stats.$SLURM_ARRAY_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/q/done.379555.$SLURM_ARRAY_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  --time 2-00:00:00 --mem-per-cpu 1900M -p cpu --mem-per-cpu 2000M --account=bbjs-delta-cpu --cpus-per-task 4  --open-mode=append -e exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/q/stats.log -o exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/q/stats.log --array 1-32 /work/nvme/bbjs/ttao3/espnet/egs2/titw/tts1/exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/q/stats.sh >>exp/24k_f5/tts_stats_raw_phn_tacotron_g2p_en_no_space/logdir/q/stats.log 2>&1
