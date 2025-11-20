# 1) Generate utt2spk & spk2utt from DISPLACE RTTMs
python3 make_spk_lists_from_rttm.py \
  --base_displace_dir /work/nvme/bbjs/ttao3/displace_audio

# 2) (Optional) Validate/fix (if you have Kaldi/ESPnet utils)
utils/fix_data_dir.sh data/displace_train
utils/validate_data_dir.sh --no-feats --no-text data/displace_train
utils/fix_data_dir.sh data/displace_dev
utils/validate_data_dir.sh --no-feats --no-text data/displace_dev
utils/fix_data_dir.sh data/displace_test
utils/validate_data_dir.sh --no-feats --no-text data/displace_test
utils/fix_data_dir.sh data/displace_dev_4dup
utils/validate_data_dir.sh --no-feats --no-text data/displace_dev
utils/fix_data_dir.sh data/displace_test_4dup
utils/validate_data_dir.sh --no-feats --no-text data/displace_test
