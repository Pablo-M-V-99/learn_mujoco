import json


def creazioneGriglia(lunghezzaGriglia, larghezzaGriglia, altezzaGriglia,  dimCella, offX, offY, offZ):
    """
    Genera la griglia cartesiana con il primo nodo nell'origine (0, 0, 0). Gli offset servono a spostare la posizione
    iniziale del primo nodo. Ciascun nodo corrisponde ad una posizione lungo la traiettoria da seguire.
    :param lunghezzaGriglia: numero di nodi della griglia lungo Y
    :param larghezzaGriglia: numero di nodi della griglia lungo X
    :param altezzaGriglia: numero di nodi della griglia lungo Z
    :param dimCella: distanza fra due nodi adiacenti
    :param offX: offset lungo X
    :param offY: offset lungo Y
    :param offZ: offset lungo Z
    """

    griglia = {}  # dizionario vuoto

    for k in range(altezzaGriglia):
        for j in range(lunghezzaGriglia):
            for i in range(larghezzaGriglia):
                chiave = f"cella_{i}_{j}_{k}"
                valore = [i * dimCella - offX, j * dimCella - offY, k * dimCella - offZ]     # coordinate spaziali della cella
                griglia[chiave] = valore

    percorso_file = f"griglia_{i+1}x{j+1}x{k+1}.json"

    with open(percorso_file, "w") as file:
        json.dump(griglia, file, indent=2)


def creazioneGrigliaRadiale():
    """
    Genera la griglia radiale. Ogni valore corrisponde ad una possibile angolazione
    """

    grigliaRad = {}

    for i in range(-180, 180+1, 1):
        chiave = f"rot_{i}"
        valore = i
        grigliaRad[chiave] = valore

    percorso_file = f"grigliaRadiale.json"

    with open(percorso_file, "w") as file:
        json.dump(grigliaRad, file, indent=2)


if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente. Usalo fornendogli in input la lunghezza della griglia,"
          " la larghezza, la distanza fra celle e gli offset in X, Y e Z.")
