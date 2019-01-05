import json
params = json.loads(input())
import optimization
result = optimization.trainRound1Optimization(params)
print(json.dumps(result))
exit(0)
