import subprocess


def main(N, script):
    for i in range(N):
        print(f"Esecuzione {i + 1} di {N}")
        subprocess.run(script, shell=True)


if __name__ == '__main__':
    N = 3                                  # numero di esecuzioni
    script = 'python muoviBandieraV3.py'           # script da eseguire N volte
    main(N, script)
