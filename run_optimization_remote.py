import json
params = json.loads(input())
import optimization
result = optimization.trainRound1Optimization(params)
print(params['$scriptToken'], flush=True)
print(json.dumps(result), flush=True)
exit(0)
