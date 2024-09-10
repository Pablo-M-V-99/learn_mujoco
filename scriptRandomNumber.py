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


def genera_numeri_casuali(dim_lista, seed):
    """
    Genera una lista di numeri casuali compresi fra 0 e 1.

    :param dim_lista: dimensione della lista.
    :param seed: il seed.
    :return: la lista di numeri casuali.
    """


    # Impostiamo il seed all'inizio dell'esperimento
    random.seed(seed)

    # Lista per conservare i numeri generati
    numeri_generati = []

    for i in range(dim_lista):
        # Generiamo numeri casuali
        numero = random.uniform(0, 1)
        numeri_generati.append(numero)

    return numeri_generati

if __name__ == "__main__":

    # print(generaNumeroCasuale(0, 100))
    print("Questo script non deve essere eseguito autonomamente.")
