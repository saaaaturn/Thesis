#!/bin/bash
# Setup script to export WORK_DIR environment variable
export WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "WORK_DIR set to: $WORK_DIR"
