import os
import sys
import subprocess
import time
import webview

def get_resource_path(relative_path):
    """ Retrouve le chemin des fichiers du projet (compatible PyInstaller) """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def demarrer_streamlit():
    dashboard_path = get_resource_path("dashboard.py")
    cmd = [
        sys.executable,
        "-m", "streamlit", "run", dashboard_path,
        "--server.headless=true",
        "--server.port=8501",
        "--global.developmentMode=false"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    # 1. Lancement du serveur Streamlit en arrière-plan
    proc = demarrer_streamlit()
    time.sleep(3)

    # 2. Ouverture de la fenêtre d'application de bureau
    webview.create_window(
        "Prédiction Ducting Atmosphérique 5G",
        "http://localhost:8501",
        width=1400,
        height=900
    )
    webview.start()

    # 3. Fermeture propre à la sortie
    proc.terminate()