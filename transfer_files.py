import subprocess
import os

try:
    subprocess.run(["rsync", "-avz", "--delete", "--checksum", "--progress", "/home/danieledinosse/GAN_Anomaly_Detection/", "ddinosse@trantor01.sns.it:/home/ddinosse/GAN_Anomaly_Detection"], check=True)
    # subprocess.run(["rsync", "-avz", "--delete", "--checksum", "--progress", "/home/danieledinosse/data/", "ddinosse@trantor01.sns.it:/home/ddinosse/data"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Rsync failed with error code {e.returncode}")