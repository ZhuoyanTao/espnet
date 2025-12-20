import re

# Step 1: Paste your avg_pos format here
raw_metrics = """
rir_room_size@meta_label: avg_pos=1.82, count=3000/3000
qwen_speech_impairment@meta_label: avg_pos=12.65, count=3000/3000
qwen_speech_clarity@meta_label: avg_pos=13.15, count=3000/3000
qwen_laughter_crying@meta_label: avg_pos=16.60, count=3000/3000
qwen_speech_purpose@meta_label: avg_pos=20.66, count=3000/3000
qwen_vocabulary_complexity@meta_label: avg_pos=22.89, count=3000/3000
qwen_speech_register@meta_label: avg_pos=24.20, count=3000/3000
qwen_speaker_gender@meta_label: avg_pos=24.23, count=3000/3000
qwen_speech_background_environment@meta_label: avg_pos=25.61, count=3000/3000
qwen_speaker_count@meta_label: avg_pos=25.67, count=3000/3000
qwen_language@meta_label: avg_pos=26.27, count=3000/3000
qwen_channel_type@meta_label: avg_pos=26.91, count=3000/3000
snr_simulation@meta_label: avg_pos=28.69, count=3000/3000
qwen_pitch_range@meta_label: avg_pos=30.58, count=3000/3000
qwen_speaking_style@meta_label: avg_pos=31.06, count=3000/3000
qwen_speech_volume_level@meta_label: avg_pos=32.27, count=3000/3000
qwen_voice_pitch@meta_label: avg_pos=32.83, count=3000/3000
qwen_speech_emotion@meta_label: avg_pos=35.83, count=3000/3000
qwen_speech_rate@meta_label: avg_pos=37.01, count=3000/3000
qwen_recording_quality@meta_label: avg_pos=37.58, count=3000/3000
qwen_speaker_age@meta_label: avg_pos=40.56, count=3000/3000
qwen_voice_type@meta_label: avg_pos=45.65, count=3000/3000
rt60@meta_label: avg_pos=51.56, count=3000/3000
noresqa_score@meta_label: avg_pos=56.03, count=3000/3000
utmos@meta_label: avg_pos=56.82, count=3000/3000
dns_overall@meta_label: avg_pos=61.46, count=3000/3000
audiobox_aesthetics_CU@meta_label: avg_pos=61.71, count=3000/3000
utmosv2@meta_label: avg_pos=61.86, count=3000/3000
nisqa_col_pred@meta_label: avg_pos=64.53, count=3000/3000
asr_match_error_rate@meta_label: avg_pos=64.57, count=3000/3000
audiobox_aesthetics_PC@meta_label: avg_pos=67.13, count=3000/3000
speech_token_distance@meta_label: avg_pos=67.24, count=3000/3000
real_language@meta_label: avg_pos=67.52, count=3000/3000
scoreq_nr@meta_label: avg_pos=67.79, count=3000/3000
nisqa_noi_pred@meta_label: avg_pos=69.98, count=3000/3000
pred_text_length@meta_label: avg_pos=70.37, count=3000/3000
plcmos@meta_label: avg_pos=70.38, count=3000/3000
ref_text_length@meta_label: avg_pos=70.80, count=3000/3000
nomad@meta_label: avg_pos=74.22, count=3000/3000
emotion_similarity@meta_label: avg_pos=74.95, count=3000/3000
spk_similarity@meta_label: avg_pos=76.88, count=3000/3000
speaking_rate@meta_label: avg_pos=77.34, count=3000/3000
audiobox_aesthetics_PQ@meta_label: avg_pos=79.31, count=3000/3000
audiobox_aesthetics_CE@meta_label: avg_pos=81.76, count=3000/3000
language@meta_label: avg_pos=84.41, count=3000/3000
singmos@meta_label: avg_pos=85.56, count=3000/3000
pam_score@meta_label: avg_pos=86.96, count=3000/3000
asvspoof_score@meta_label: avg_pos=87.00, count=3000/3000
nisqa_loud_pred@meta_label: avg_pos=87.66, count=3000/3000
se_sdr@meta_label: avg_pos=89.04, count=3000/3000
speech_bert@meta_label: avg_pos=90.82, count=3000/3000
dns_p808@meta_label: avg_pos=93.10, count=3000/3000
scoreq_ref@meta_label: avg_pos=101.18, count=3000/3000
sheet_ssqa@meta_label: avg_pos=104.14, count=3000/3000
se_sar@meta_label: avg_pos=106.62, count=3000/3000
se_si_snr@meta_label: avg_pos=109.35, count=3000/3000
se_ci_sdr@meta_label: avg_pos=109.49, count=3000/3000
nisqa_dis_pred@meta_label: avg_pos=110.53, count=3000/3000
nisqa_mos_pred@meta_label: avg_pos=112.50, count=3000/3000
speech_bleu@meta_label: avg_pos=115.49, count=3000/3000
pysepm_wss@meta_label: avg_pos=116.23, count=3000/3000
pysepm_fwsegsnr@meta_label: avg_pos=118.87, count=3000/3000
pysepm_cd@meta_label: avg_pos=120.58, count=3000/3000
f0corr@meta_label: avg_pos=125.41, count=3000/3000
sdr@meta_label: avg_pos=127.53, count=3000/3000
sar@meta_label: avg_pos=128.91, count=3000/3000
pysepm_c_bak@meta_label: avg_pos=131.09, count=3000/3000
pysepm_llr@meta_label: avg_pos=131.68, count=3000/3000
pysepm_c_ovl@meta_label: avg_pos=132.49, count=3000/3000
pysepm_csii_high@meta_label: avg_pos=133.82, count=3000/3000
stoi@meta_label: avg_pos=134.80, count=3000/3000
pysepm_csii_low@meta_label: avg_pos=135.71, count=3000/3000
pysepm_ncm@meta_label: avg_pos=135.72, count=3000/3000
pysepm_c_sig@meta_label: avg_pos=136.42, count=3000/3000
f0rmse@meta_label: avg_pos=137.22, count=3000/3000
mcd@meta_label: avg_pos=139.50, count=3000/3000
visqol@meta_label: avg_pos=141.33, count=3000/3000
ci_sdr@meta_label: avg_pos=146.56, count=3000/3000
srmr@meta_label: avg_pos=148.52, count=3000/3000
pesq@meta_label: avg_pos=150.61, count=3000/3000
pysepm_csii_mid@meta_label: avg_pos=154.23, count=3000/3000
si_snr@meta_label: avg_pos=154.99, count=3000/3000
urgent_mos@meta_label: avg_pos=163.41, count=3000/3000
wer@meta_label: avg_pos=166.34, count=3000/3000
cer@meta_label: avg_pos=167.26, count=3000/3000
nisqa_real_mos@meta_label: avg_pos=170.62, count=3000/3000
voicemos_real_mos@meta_label: avg_pos=171.38, count=3000/3000
"""

# Step 2: Mapping from raw to canonical names
name_mapping = {
    "Q-Gender": "qwen_speaker_gender",
    "Q-SpeechImpariment": "qwen_speech_impairment",
    "Q-SpeakingStyle": "qwen_speaking_style",
    "Q-EnvQuality": "qwen_recording_quality",
    "Q-PitchRange": "qwen_pitch_range",
    "Q-VocComplexity": "qwen_vocabulary_complexity",
    "Q-VolumeLevel": "qwen_speech_volume_level",
    "RealLanguage": "real_language",
    "Q-ContentRegister": "qwen_speech_register",
    "SRMR": "srmr",
    "SpoofS": "asvspoof_score",
    "NISQA-NOI": "nisqa_noi_pred",
    "Q-Emotion": "qwen_speech_emotion",
    "AA-PC": "audiobox_aesthetics_PC",
    "Q-Background": "qwen_speech_background_environment",
    "AA-PQ": "audiobox_aesthetics_PQ",
    "Q-ChannelType": "qwen_channel_type",
    "LID": "language",
    "Q-Clarity": "qwen_speech_clarity",
    "SE-CI-SDR": "se_ci_sdr",
    "DNSMOSP.835": "dns_overall",
    "SWR/SCR": "speaking_rate",  # please confirm this alias
    "Q-Purpose": "qwen_speech_purpose",
    "WER": "wer",
    "Q-VoiceType": "qwen_voice_type",
    "SingMOS": "singmos",
    "SE-SI-SNR": "se_si_snr",
    "Q-Lang": "qwen_language",
    "Q-SpeechRate": "qwen_speech_rate",
    "SCOREQ": "scoreq_nr",
    "NISQA-COL": "nisqa_col_pred",
    "Q-EmotionalVocalization": "qwen_laughter_crying",
    "NISQA-LOUD": "nisqa_loud_pred",
    "Q-SpeakerCount": "qwen_speaker_count",
    "Q-Age": "qwen_speaker_age",
    "PAM": "pam_score",
    "UTMOS": "utmos",
    "AA-CE": "audiobox_aesthetics_CE",
    "NISQA-MOS": "nisqa_mos_pred",
    "DNSMOSP.808": "dns_p808",
    "SSQA": "ssqa",
    "PLCMOS": "plcmos",
    "Q-Pitch": "qwen_voice_pitch",
    "AA-CU": "audiobox_aesthetics_CU",
    "CER": "cer",
    "SE-SDR": "se_sdr",
    "UTMOSv2": "utmosv2",
    "NISQA-DIS": "nisqa_dis_pred",
    "CI-SDR": "ci_sdr",
    "D-Distance": "speech_token_distance",
    "STOI": "stoi",
    "SE-SAR": "se_sar",
    "SDR": "sdr",
    "SI-SNR": "si_snr",
    "MCD": "mcd",
    "D-BERT": "speech_bert",
    "F0Corr": "f0corr",
    "F0RMSE": "f0rmse",
    "SPK-SIM": "spk_similarity",
    "D-BLEU": "speech_bleu",
    "PESQ": "pesq",
    "URGENT MOS": "urgent_mos",
    "SAR": "sar",
    "CD": "cd",
    "WSS": "wss",
    "LLR": "llr",
    "EMO-SIM": "emo_sim",
    "NCM": "ncm",
    "Covl": "covl",
    "Reference Text Length": "reference_text_length",
    "VISQOL": "visqol",
    "Csig": "csig",
    "CSII-MID": "csii_mid",
    "Cbak": "cbak",
    "CSII-HIGH": "csii_high",
    "SCOREQ w. Ref.": "scoreq_with_ref",
    "ASR-Mismatch": "asr_mismatch",
    "CSII-LOW": "csii_low",
    "NOMAD": "nomad",
    "RIR Room Size": "rir_room_size",
    "Noresqa": "noresqa",
    "FWSEGSNR": "fwsegsnr",
    "RT60": "rt60",
    "Predicted Text Length": "predicted_text_length",
    "SNR Simulation": "snr_simulation",
    "NISQAReal MOS": "nisqa_real_mos",
    "VoiceMOSReal MOS": "voicemos_real_mos"
}


# Step 3: Parse the metric order string (use your existing format here)
final_format = """
3/0/1/{Q-SpeechImpariment},
6/0/2/{Q-SpeakingStyle},
9/0/3/{Q-EnvQuality},
12/0/4/{Q-PitchRange},
15/0/5/{Q-VocComplexity},
18/0/6/{Q-VolumeLevel},
21/0/7/{RealLanguage},
24/0/8/{Q-ContentRegister},
27/0/9/{SRMR},
0/-1/10/{SpoofS},
3/-1/11/{NISQA-NOI},
6/-1/12/{Q-Emotion},
9/-1/13/{AA-PC},
12/-1/14/{Q-Background},
15/-1/15/{AA-PQ},
18/-1/16/{Q-ChannelType},
21/-1/17/{LID},
24/-1/18/{Q-Clarity},
27/-1/19/{SE-CI-SDR},
0/-2/20/{DNSMOSP.835},
3/-2/21/{SWR/SCR},
6/-2/22/{Q-Purpose},
9/-2/23/{WER},
12/-2/24/{Q-VoiceType},
15/-2/25/{SingMOS},
18/-2/26/{SE-SI-SNR},
21/-2/27/{Q-Lang},
24/-2/28/{Q-SpeechRate},
27/-2/29/{SCOREQ},
0/-3/30/{NISQA-COL},
3/-3/31/{Q-EmotionalVocalization},
6/-3/32/{NISQA-LOUD},
9/-3/33/{Q-SpeakerCount},
12/-3/34/{Q-Age},
15/-3/35/{PAM},
18/-3/36/{UTMOS},
21/-3/37/{AA-CE},
24/-3/38/{NISQA-MOS},
27/-3/39/{DNSMOSP.808},
0/-4/40/{SSQA},
3/-4/41/{PLCMOS},
6/-4/42/{Q-Pitch},
9/-4/43/{AA-CU},
12/-4/44/{CER},
15/-4/45/{SE-SDR},
18/-4/46/{UTMOSv2},
21/-4/47/{NISQA-DIS},
24/-4/48/{CI-SDR},
27/-4/49/{D-Distance},
0/-5/50/{STOI},
3/-5/51/{SE-SAR},
6/-5/52/{SDR},
9/-5/53/{SI-SNR},
12/-5/54/{MCD},
15/-5/55/{D-BERT},
18/-5/56/{F0Corr},
21/-5/57/{F0RMSE},
24/-5/58/{SPK-SIM},
27/-5/59/{D-BLEU},
0/-6/60/{PESQ},
3/-6/61/{URGENT MOS},
6/-6/62/{SAR},
9/-6/63/{CD},
12/-6/64/{WSS},
15/-6/65/{LLR},
18/-6/66/{EMO-SIM},
21/-6/67/{NCM},
24/-6/68/{Covl},
27/-6/69/{Reference Text Length},
0/-7/70/{VISQOL},
3/-7/71/{Csig},
6/-7/72/{CSII-MID},
9/-7/73/{Cbak},
12/-7/74/{CSII-HIGH},
15/-7/75/{SCOREQ w. Ref.},
18/-7/76/{ASR-Mismatch},
21/-7/77/{CSII-LOW},
24/-7/78/{NOMAD},
27/-7/79/{RIR Room Size},
0/-8/80/{Noresqa},
3/-8/81/{FWSEGSNR},
6/-8/82/{RT60},
9/-8/83/{Predicted Text Length},
12/-8/84/{SNR Simulation},
15/-8/85/{NISQAReal MOS},
18/-8/86/{VoiceMOSReal MOS}
"""  # Truncated here for brevity; you should use the full input


# === PARSE raw avg_pos into a rank dict ===
metric_rank = {}
for line in raw_metrics.strip().splitlines():
    name, pos = re.match(r"([^@]+)@meta_label: avg_pos=([\d.]+)", line).groups()
    metric_rank[name.strip()] = float(pos)

# === PROCESS FINAL FORMAT ===

def canonical_to_raw(metric):
    return name_mapping.get(metric.strip(), metric.strip())

# Replace each {Canonical} with {Raw}, keeping line order
output = []
for line in final_format.strip().splitlines():
    match = re.search(r"\{([^}]+)\}", line)
    if match:
        canonical = match.group(1)
        raw = canonical_to_raw(canonical)
        updated = line.replace(f"{{{canonical}}}", f"{{{raw}}}")
        output.append(updated)
    else:
        output.append(line)

# === OUTPUT ===
print("\n".join(output))
