import json
import os
import datetime


def saveLabels(angles, poses, folder):

    dizionario = {}  # dizionario vuoto
    valore = []
    chiave = f"labels"

    for i in range(len(angles)):
        valore.append([poses[i][0], poses[i][1], poses[i][2], angles[i][0], angles[i][1], angles[i][2]])

    dizionario[chiave] = valore

    percorso_file = f"{folder}/labels.json"
    with open(percorso_file, "w") as file:
        json.dump(dizionario, file, indent=2)


def saveParameters(mass, radius, damping, poisson, young, thickness, pos, dimension, spacing, posizione_manoDx,
                   posizione_manoSx, seed, folder):

    dizionario = {}  # dizionario vuoto
    valore = []

    dizionario["mass"] = mass
    dizionario["radius"] = radius
    dizionario["damping"] = damping
    dizionario["poisson"] = poisson
    dizionario["young"] = young
    dizionario["thickness"] = thickness
    dizionario["pos"] = pos
    dizionario["dimension"] = dimension
    dizionario["spacing"] = spacing
    dizionario["posizione_manoDx"] = posizione_manoDx
    dizionario["posizione_manoSx"] = posizione_manoSx
    dizionario["seed"] = seed

    percorso_file = f"{folder}/parameters.json"
    with open(percorso_file, "w") as file:
        json.dump(dizionario, file, indent=2)


if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
