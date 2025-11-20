import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import re

# Step 1: Paste your raw metric string here
raw_metrics = """
qwen_speaker_gender@meta_label: avg_pos=16.50, count=11400/11400
qwen_speech_impairment@meta_label: avg_pos=20.35, count=11400/11400
qwen_speaking_style@meta_label: avg_pos=21.47, count=11400/11400
qwen_recording_quality@meta_label: avg_pos=24.58, count=11400/11400
qwen_pitch_range@meta_label: avg_pos=31.61, count=11400/11400
qwen_vocabulary_complexity@meta_label: avg_pos=31.86, count=11400/11400
qwen_speech_volume_level@meta_label: avg_pos=33.35, count=11400/11400
real_language@meta_label: avg_pos=36.36, count=11400/11400
qwen_speech_register@meta_label: avg_pos=37.95, count=11400/11400
srmr@meta_label: avg_pos=38.04, count=11400/11400
asvspoof_score@meta_label: avg_pos=38.92, count=11400/11400
nisqa_noi_pred@meta_label: avg_pos=42.05, count=11400/11400
qwen_speech_emotion@meta_label: avg_pos=42.35, count=11400/11400
audiobox_aesthetics_PC@meta_label: avg_pos=45.00, count=11400/11400
qwen_speech_background_environment@meta_label: avg_pos=45.81, count=11400/11400
audiobox_aesthetics_PQ@meta_label: avg_pos=46.16, count=11400/11400
qwen_channel_type@meta_label: avg_pos=47.31, count=11400/11400
language@meta_label: avg_pos=49.13, count=11400/11400
qwen_speech_clarity@meta_label: avg_pos=49.30, count=11400/11400
se_ci_sdr@meta_label: avg_pos=50.76, count=11400/11400
dns_overall@meta_label: avg_pos=50.97, count=11400/11400
speaking_rate@meta_label: avg_pos=52.43, count=11400/11400
qwen_speech_purpose@meta_label: avg_pos=53.46, count=11400/11400
wer@meta_label: avg_pos=53.66, count=11400/11400
qwen_voice_type@meta_label: avg_pos=54.42, count=11400/11400
singmos@meta_label: avg_pos=54.78, count=11400/11400
se_si_snr@meta_label: avg_pos=55.01, count=11400/11400
qwen_language@meta_label: avg_pos=55.99, count=11400/11400
qwen_speech_rate@meta_label: avg_pos=57.39, count=11400/11400
scoreq_nr@meta_label: avg_pos=57.77, count=11400/11400
nisqa_col_pred@meta_label: avg_pos=60.58, count=11400/11400
qwen_laughter_crying@meta_label: avg_pos=60.64, count=11400/11400
nisqa_loud_pred@meta_label: avg_pos=61.67, count=11400/11400
qwen_speaker_count@meta_label: avg_pos=61.94, count=11400/11400
qwen_speaker_age@meta_label: avg_pos=62.98, count=11400/11400
pam_score@meta_label: avg_pos=63.10, count=11400/11400
utmos@meta_label: avg_pos=64.52, count=11400/11400
audiobox_aesthetics_CE@meta_label: avg_pos=64.53, count=11400/11400
nisqa_mos_pred@meta_label: avg_pos=66.34, count=11400/11400
dns_p808@meta_label: avg_pos=67.91, count=11400/11400
sheet_ssqa@meta_label: avg_pos=68.65, count=11400/11400
plcmos@meta_label: avg_pos=68.68, count=11400/11400
qwen_voice_pitch@meta_label: avg_pos=72.34, count=11400/11400
audiobox_aesthetics_CU@meta_label: avg_pos=72.43, count=11400/11400
cer@meta_label: avg_pos=72.71, count=11400/11400
se_sdr@meta_label: avg_pos=74.95, count=11400/11400
utmosv2@meta_label: avg_pos=79.76, count=11400/11400
nisqa_dis_pred@meta_label: avg_pos=82.15, count=11400/11400
ci_sdr@meta_label: avg_pos=82.25, count=11400/11400
speech_token_distance@meta_label: avg_pos=83.90, count=11400/11400
stoi@meta_label: avg_pos=84.32, count=11400/11400
se_sar@meta_label: avg_pos=85.62, count=11400/11400
sdr@meta_label: avg_pos=86.10, count=11400/11400
si_snr@meta_label: avg_pos=87.77, count=11400/11400
mcd@meta_label: avg_pos=89.90, count=11400/11400
speech_bert@meta_label: avg_pos=90.61, count=11400/11400
f0corr@meta_label: avg_pos=93.14, count=11400/11400
f0rmse@meta_label: avg_pos=94.55, count=11400/11400
spk_similarity@meta_label: avg_pos=98.41, count=11400/11400
speech_bleu@meta_label: avg_pos=99.28, count=11400/11400
pesq@meta_label: avg_pos=102.43, count=11400/11400
urgent_mos@meta_label: avg_pos=105.44, count=11400/11400
sar@meta_label: avg_pos=108.58, count=11400/11400
pysepm_cd@meta_label: avg_pos=129.25, count=11400/11400
pysepm_wss@meta_label: avg_pos=131.61, count=11400/11400
pysepm_llr@meta_label: avg_pos=133.28, count=11400/11400
emotion_similarity@meta_label: avg_pos=135.85, count=11400/11400
pysepm_ncm@meta_label: avg_pos=136.52, count=11400/11400
pysepm_c_ovl@meta_label: avg_pos=137.29, count=11400/11400
ref_text_length@meta_label: avg_pos=138.62, count=11400/11400
visqol@meta_label: avg_pos=138.87, count=11400/11400
pysepm_c_sig@meta_label: avg_pos=142.02, count=11400/11400
pysepm_csii_mid@meta_label: avg_pos=144.53, count=11400/11400
pysepm_c_bak@meta_label: avg_pos=144.96, count=11400/11400
pysepm_csii_high@meta_label: avg_pos=146.82, count=11400/11400
scoreq_ref@meta_label: avg_pos=147.95, count=11400/11400
asr_match_error_rate@meta_label: avg_pos=148.58, count=11400/11400
pysepm_csii_low@meta_label: avg_pos=152.09, count=11400/11400
nomad@meta_label: avg_pos=153.53, count=11400/11400
rir_room_size@meta_label: avg_pos=156.55, count=11400/11400
noresqa_score@meta_label: avg_pos=159.11, count=11400/11400
pysepm_fwsegsnr@meta_label: avg_pos=161.97, count=11400/11400
rt60@meta_label: avg_pos=163.30, count=11400/11400
pred_text_length@meta_label: avg_pos=163.37, count=11400/11400
snr_simulation@meta_label: avg_pos=163.52, count=11400/11400
nisqa_real_mos@meta_label: avg_pos=167.91, count=11400/11400
voicemos_real_mos@meta_label: avg_pos=171.58, count=11400/11400
"""

# Step 2: Preprocess the text into list of tuples
lines = raw_metrics.strip().split('\n')
metric_data = []

for line in lines:
    match = re.match(r"([^:]+): avg_pos=([\d.]+)", line)
    if match:
        name = match.group(1).strip()
        value = float(match.group(2))
        metric_data.append((name, value))

# Step 3: Convert to DataFrame and sort
df = pd.DataFrame(metric_data, columns=["Metric", "Avg Pos"]).sort_values(by="Avg Pos").reset_index(drop=True)

# Step 4: Generate colors from red to blue
def get_color_map(values, cmap_name='RdYlBu_r'):
    norm = plt.Normalize(min(values), max(values))
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(norm(v))) for v in values]

colors = get_color_map(df["Avg Pos"])

# Step 5: Output TikZ code
print("\\begin{figure}[ht]")
print("\\centering")
print("\\caption{Visualization of Metric Order via Color Blocks (Red = Early, Blue = Late)}")
print("\\begin{tikzpicture}")

for i, (row, color) in enumerate(zip(df.itertuples(), colors)):
    x = i % 20
    y = -(i // 20)
    print(f"  \\filldraw[fill={color}, draw=black] ({x}, {y}) rectangle ++(1, 1);")
    print(f"  \\node[rotate=90, anchor=west, font=\\tiny] at ({x + 0.5}, {y}) {{{row.Metric}}};")

print("\\end{tikzpicture}")
print("\\end{figure}")
