#!/bin/bash
cd /work/nvme/bbjs/ttao3/espnet/egs2/globe/tts1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python3 local/data_prep.py --train_set train --dev_set val --test_set test --hf_repo MushanW/GLOBE_V2 --dest_path data --jobs 4 
EOF
) >logs/data_prep/data_prep.$SLURM_ARRAY_TASK_ID.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>logs/data_prep/data_prep.$SLURM_ARRAY_TASK_ID.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( python3 local/data_prep.py --train_set train --dev_set val --test_set test --hf_repo MushanW/GLOBE_V2 --dest_path data --jobs 4  ) &>>logs/data_prep/data_prep.$SLURM_ARRAY_TASK_ID.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>logs/data_prep/data_prep.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: end_time=$time2 >>logs/data_prep/data_prep.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>logs/data_prep/data_prep.$SLURM_ARRAY_TASK_ID.log
echo '#' Finished at `date` with status $ret >>logs/data_prep/data_prep.$SLURM_ARRAY_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch logs/data_prep/q/done.373615.$SLURM_ARRAY_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  -p cpu --mem-per-cpu 2000M --account=bbjs-delta-cpu --time 2-00:00:00 --cpus-per-task 4 --mem-per-cpu 1900M  --open-mode=append -e logs/data_prep/q/data_prep.log -o logs/data_prep/q/data_prep.log --array 1-1 /work/nvme/bbjs/ttao3/espnet/egs2/globe/tts1/logs/data_prep/q/data_prep.sh >>logs/data_prep/q/data_prep.log 2>&1
