import subprocess
import multiprocessing
import sys


scripts = [
    "lr.py",
    "rf.py",
    "gb.py"
]


def run_script(script):
    print(f"Starting {script}")
    subprocess.run([sys.executable, script])
    print(f"Finished {script}")


if __name__ == "__main__":

    processes = []

    for script in scripts:
        p = multiprocessing.Process(target=run_script, args=(script,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nAll MLflow experiments completed.")