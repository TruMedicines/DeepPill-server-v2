import json
import sys
import psutil
params = json.loads(input())
import optimization
result = optimization.trainRound1Optimization(params)
print(params['$scriptToken'], flush=True)
print(json.dumps(result), flush=True)

current_process = psutil.Process()
children = current_process.children(recursive=True)
for child in children:
    child.kill()
sys.exit(0)