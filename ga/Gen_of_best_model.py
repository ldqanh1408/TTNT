from ga.ga_ai import GeneticAlgorithm
net, meta = GeneticAlgorithm.load_best("model/ga_best.pkl")
weights = net.get_flat_weights()
print(weights)  # mảng 37,379 số float64
print(weights.shape)  # (37379,)