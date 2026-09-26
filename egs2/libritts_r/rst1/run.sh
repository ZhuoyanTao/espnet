#!/usr/bin/env bash
# LibriTTS-R + EARS + VCTK restoration recipe (Sidon reproduction). The
# corpus-specific parts are local/data.sh and conf/; everything else is the
# shared driver egs2/TEMPLATE/rst1/rst.sh, whose defaults are this recipe's.
set -euo pipefail

./rst.sh \
    --test_sets "test-clean test-other" \
    "$@"
