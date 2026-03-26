import torch
import yaml
from argparse import Namespace
from espnet2.tasks.universa import UniversaTask

############################################
# 1. Load Config Properly
############################################

CONFIG_PATH = "conf/train_aruniversa_prefix.yaml"
CHECKPOINT_PATH = "exp/universa_universa_ar_overall_base_token_wavlm_large/valid.loss.best.prefix_expanded.pth"

print("Loading config...")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

args = Namespace(**cfg)

# Inject required runtime arguments
args.metric2id = "dump/raw/prefix_train/metric2id"
args.metric2type = "dump/raw/prefix_train/metric2type"
args.use_preprocessor = True
args.ngpu = 0
args.tokenize_numerical_metric = True
args.token_list = None
args.metric2id = "dump/raw/prefix_train/metric2id"
args.metric2type = "dump/raw/prefix_train/metric2type"

args.metric_token_info = "data/token_list/metric_500_percentile_overall_base_w-numerical/tokens.json"
args.metric_token_pad_value = 0
args.metric_pad_value = -100

args.use_preprocessor = True
args.tokenize_numerical_metric = True
args.sequential_metric = True
args.randomize_sequential_metric = True

args.token_list = None
args.ngpu = 0


############################################
# 2. Build Model
############################################

print("Building model...")
model = UniversaTask.build_model(args)
model.eval()

print("Loading checkpoint...")
state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)

print("Model ready.\n")

############################################
# 3. Dummy Inputs
############################################

B = 1
T = 16000 * 4  # 4 seconds fake audio

dummy_audio = torch.randn(B, T)
dummy_ref_audio = torch.randn(B, T)
zero_ref_audio = torch.zeros(B, T)

dummy_metrics = torch.randint(0, 10, (B, 5))

############################################
# 4. Forward Tests
############################################

def run_test(name, ref_audio_value):
    print(f"Running test: {name}")

    batch = {
        "audio": dummy_audio,
        "metrics": dummy_metrics,
    }

    if ref_audio_value is not None:
        batch["ref_audio"] = ref_audio_value

    try:
        with torch.no_grad():
            out = model(**batch)

        if isinstance(out, tuple):
            out = out[0]

        print("  ✔ Success")
        print("  Output shape:", out.shape)
        return out

    except Exception as e:
        print("  ✘ Failed:", e)
        return None


out_real = run_test("Real Ref Audio", dummy_ref_audio)
out_zero = run_test("Zero Ref Audio", zero_ref_audio)
out_none = run_test("No Ref Audio", None)

############################################
# 5. Compare Outputs
############################################

if out_real is not None and out_zero is not None:
    diff = torch.norm(out_real - out_zero)
    print("\nDifference (real vs zero ref):", diff.item())

if out_real is not None and out_none is not None:
    diff2 = torch.norm(out_real - out_none)
    print("Difference (real vs no ref):", diff2.item())

print("\nDone.")
