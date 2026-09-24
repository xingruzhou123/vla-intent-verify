#!/usr/bin/env bash
# Download key related-work PDFs into docs/papers/ (not committed to git).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p papers
declare -A PAPERS=(
  [redvla_simplevla_guard]=2604.22591
  [safe]=2506.09937
  [hazardarena]=2604.12447
  [safecast]=2608.04246
  [vla_fail]=2606.21386
  [failurespot]=2609.04277
  [tri_info]=2606.19998
  [safe_dict]=2602.01834
  [pre_vla]=2605.22446
  [actfovea]=2607.29169
  [vla_commonsense_probing]=2606.19297
  [vla_sae]=2603.19183
  [vla_observe_control]=2603.05487
  [foresight]=2606.23085
)
for name in "${!PAPERS[@]}"; do
  id=${PAPERS[$name]}
  out="papers/${name}_${id}.pdf"
  [ -f "$out" ] && continue
  echo "Fetching $name ($id)"
  curl -sL "https://arxiv.org/pdf/${id}" -o "$out"
  sleep 3   # be polite to arXiv
done
echo "Done."
