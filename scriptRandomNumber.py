import random

def generaNumeroCasuale(minimo, massimo, intero=True):
    """
    Genera un numero casuale.

    :param minimo: Il valore minimo del range.
    :param massimo: Il valore massimo del range.
    :param intero: Se True, genera un numero intero. Se False, genera un numero decimale.
    :return: Un numero casuale all'interno del range specificato.
    """
    if intero:
        return random.randint(minimo, massimo)
    else:
        return random.uniform(minimo, massimo)

if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
