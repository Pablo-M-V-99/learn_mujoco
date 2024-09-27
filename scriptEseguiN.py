import subprocess
import datetime


def main(N, script):
    for i in range(N):
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"{current_time}: Esecuzione {i + 1} di {N}")
        subprocess.run(script, shell=True)


if __name__ == '__main__':
    N = 10                                  # numero di esecuzioni
    script = 'python muoviBandieraV3.py'    # script da eseguire N volte
    main(N, script)
