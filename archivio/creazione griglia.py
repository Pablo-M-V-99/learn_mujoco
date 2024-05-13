import json

lunghezzaGriglia = 0.3
larghezzaGriglia = 0.4
dist = 0.01         # distanza celle

lunghezzaBandiera = 0.9
larghezzaBandiera = 0.4
xOffset = 0.2
yOffset = 0.45
zOffset = 0

griglia = {}        # dizionario vuoto
j = 0
i = 0

for j in range(int(lunghezzaGriglia * 100) + 1):
    for i in range(int(larghezzaGriglia * 100) + 1):
        chiave = f"cella_{i}_{j}"
        valore = [i * dist - xOffset, j * dist - yOffset, 1.5- zOffset]
        griglia[chiave] = valore

percorso_file = f"griglia_{i}x{j}.json"

with open(percorso_file, "w") as file:
    json.dump(griglia, file, indent=2)
