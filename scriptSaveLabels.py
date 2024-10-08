import json


def saveLabels(angles, poses):

    dizionario = {}  # dizionario vuoto
    valore = []
    chiave = f"labels"

    i = 0
    for i in range(len(angles)):
         valore.append([poses[i][0], poses[i][1], poses[i][2], angles[i][0], angles[i][1], angles[i][2]])

    dizionario[chiave] = valore

    percorso_file = f"labels.json"
    with open(percorso_file, "w") as file:
        json.dump(dizionario, file, indent=2)


if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
