#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [futurekv-train-two-stage arguments...]" >&2
  exit 2
fi

futurekv-train-two-stage "$@"
