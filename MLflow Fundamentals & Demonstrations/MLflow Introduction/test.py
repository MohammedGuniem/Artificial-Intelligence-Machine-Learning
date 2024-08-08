hidden_layer_sizes = ["50", "100", "50|50", "100|50|25"]
print(hidden_layer_sizes)
print(type(hidden_layer_sizes[0]))



hidden_layer_sizes = [tuple(map(int, sizes.split("|"))) for sizes in hidden_layer_sizes]

print(hidden_layer_sizes)
print(type(hidden_layer_sizes[0]))