import torch

# INPUT_CKPT = "exp/universa_universa_ar_overall_base_token_wavlm_large/valid.loss.best.prefix_32926.pth"
# OUTPUT_CKPT = "exp/universa_universa_ar_overall_base_token_wavlm_large/valid.loss.best.prefix_32930.pth"
INPUT_CKPT = "/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/exp/universa_universa_ar_overall_base_token_wavlm_large/valid.loss.best.prefix_32934.pth"
OUTPUT_CKPT = "/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/exp/universa_universa_ar_overall_base_token_wavlm_large/valid.loss.best.prefix_65828.pth"

OLD_VOCAB = 32934
NEW_VOCAB = 65828   # +4
ADD_TOKENS = NEW_VOCAB - OLD_VOCAB

print("Loading checkpoint:", INPUT_CKPT)
state_dict = torch.load(INPUT_CKPT, map_location="cpu")

# ---- Embedding ----
embed_key = "universa.decoder.embed.0.weight"
out_w_key = "universa.decoder.output_layer.weight"
out_b_key = "universa.decoder.output_layer.bias"

old_embed = state_dict[embed_key]
old_out_w = state_dict[out_w_key]
old_out_b = state_dict[out_b_key]

print("Old embed shape:", old_embed.shape)

assert old_embed.shape[0] == OLD_VOCAB

# Mean initialization (safe + stable)
mean_embed = old_embed.mean(dim=0, keepdim=True)
mean_out_w = old_out_w.mean(dim=0, keepdim=True)
mean_out_b = old_out_b.mean().unsqueeze(0)

new_embed = torch.cat([old_embed, mean_embed.repeat(ADD_TOKENS, 1)], dim=0)
new_out_w = torch.cat([old_out_w, mean_out_w.repeat(ADD_TOKENS, 1)], dim=0)
new_out_b = torch.cat([old_out_b, mean_out_b.repeat(ADD_TOKENS)], dim=0)

state_dict[embed_key] = new_embed
state_dict[out_w_key] = new_out_w
state_dict[out_b_key] = new_out_b

print("New embed shape:", new_embed.shape)

torch.save(state_dict, OUTPUT_CKPT)

print("Saved expanded checkpoint to:", OUTPUT_CKPT)