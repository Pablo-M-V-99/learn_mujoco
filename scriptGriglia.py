import json


def creazioneGriglia(lunghezzaGriglia, larghezzaGriglia, altezzaGriglia,  dimCella, offX, offY, offZ):

    i = 0
    j = 0
    k = 0
    griglia = {}  # dizionario vuoto

    for k in range(int(altezzaGriglia * 100) + 1):
        for j in range(int(lunghezzaGriglia * 100) + 1):
            for i in range(int(larghezzaGriglia * 100) + 1):
                chiave = f"cella_{i}_{j}_{k}"
                valore = [i * dimCella - offX, j * dimCella - offY, k * dimCella - offZ + 1.5]     # coordinate spaziali della cella
                griglia[chiave] = valore

    percorso_file = f"griglia_{i}x{j}x{k}.json"

    with open(percorso_file, "w") as file:
        json.dump(griglia, file, indent=2)


if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente. Usalo fornendogli in input la lunghezza della griglia,"
          " la larghezza, la distanza fra celle e gli offset in X, Y e Z.")
