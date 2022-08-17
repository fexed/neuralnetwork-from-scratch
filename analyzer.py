import sys, ast


filename = sys.argv[1]
res = []
with open(filename, 'r') as f:
    for line in f:
        try:
            lst = ast.literal_eval(line)
            res.append(lst)
        except ValueError:
            continue

print("***BEST RESULTS***")
print("Best metric")
bestMetric = sorted(res, key=lambda x: x[1])[:100]
for r in bestMetric[:10]:
    print(r)
print("\n")
print("Lowest std dev")
lowestStdDev = sorted(res, key=lambda x: x[2])[:100]
for r in lowestStdDev[:10]:
    print(r)

bestModels = []
for elem in bestMetric:
    for tgt in lowestStdDev:
        if elem[0] == tgt[0]:
            bestModels.append(elem)
            break

print("\n")
print("Best models")  # common between the bestMetric and lowestStdDev
if len(bestModels) == 0: print("No best models")
else:
    for r in bestModels[:5]:
        print(r)