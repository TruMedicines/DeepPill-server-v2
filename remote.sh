#!/usr/bin/env bash
cd ~/eb-pill-match/
source venv/bin/activate
killall python3 -s 9
python3 run_optimization_remote.py