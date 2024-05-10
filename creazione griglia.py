import json

lunghezzaGriglia = 0.3
larghezzaGriglia = 0.45
dist = 0.01         # distanza celle

griglia = {}        # dizionario vuoto
j = 0
i = 0
for j in range(int(lunghezzaGriglia * 100) + 1):
    for i in range(int(larghezzaGriglia * 100) + 1):  # i parte da 0
        chiave = f"cella[{i},{j}]"
        valore = [i * dist, j * dist, 1.5]
        griglia[chiave] = valore

percorso_file = f"griglia_{i}x{j}.json"

with open(percorso_file, "w") as file:
    json.dump(griglia, file, indent=2)




