from cx_Freeze import setup, Executable

# Specifica le dipendenze e i moduli da includere
include_files = []

build_exe_options = {
    "packages": ["os", "kivy", "cv2", "numpy", "PIL", "deepface"],
    "include_files": include_files,
}

# Definisci l'eseguibile
executables = [
    Executable('app.py', base=None)  # 'base=None' per creare un eseguibile standalone
]

# Configura il setup
setup(
    name="YourAppName",
    version="0.1",
    description="Description of your app",
    options={"build_exe": build_exe_options},
    executables=executables
)