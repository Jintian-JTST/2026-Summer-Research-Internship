import subprocess

def run(cmd):
    subprocess.run(
        cmd,
        shell=True,
        text=True
    )

if __name__ == "__main__":
    run("python generation.py")
    run("python analysis.py")
    run("python real_analysis.py")
    run("python together.py")