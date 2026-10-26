#!/usr/bin/env bash
# Downloads the open-access arXiv papers used as the RAG corpus into data/.
set -euo pipefail
cd "$(dirname "$0")/.." && mkdir -p data
while read -r id name; do
  [ -f "data/$name.pdf" ] && continue
  curl -sSL --max-time 120 -A "rag-project/0.1" -o "data/$name.pdf" "https://arxiv.org/pdf/$id"
  sleep 3
done <<'LIST'
2005.11401 rag_lewis_2020
2312.10997 rag_survey_gao_2023
2402.13178 medrag_benchmark_2024
2212.13138 med_palm_2022
2305.09617 med_palm2_2023
1505.04597 unet_2015
2304.12306 medsam_2023
1711.05225 chexnet_2017
2303.13375 gpt4_medical_challenge_2023
2311.16452 medprompt_2023
LIST
echo "Downloaded to data/"
