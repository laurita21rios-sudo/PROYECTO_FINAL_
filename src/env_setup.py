import os
import sys
from pathlib import Path

def is_kaggle():
    """Verifica de forma estricta si el runtime actual es Kaggle Kernels."""
    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_DOCKER_IMAGE" in os.environ
        or (os.name != "nt" and Path("/kaggle/working").is_dir())
    )

def is_colab():
    """Verifica de forma estricta si el runtime actual es Google Colab."""
    if "google.colab" in sys.modules:
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ

def get_env_paths():
    """
    Detecta automáticamente el entorno de ejecución (Kaggle, Google Colab o Local)
    y retorna las rutas clave para garantizar portabilidad sin rutas absolutas fijas.
    """
    # 1. Detección de Kaggle
    if is_kaggle():
        env = "kaggle"
        work_root = Path("/kaggle/working")
        
        # Buscar el dataset en /kaggle/input
        kaggle_input = Path("/kaggle/input")
        dataset_candidates = [
            kaggle_input / "indian-diabetic-retinopathy-image-dataset-idrid2",
            kaggle_input / "idrid2",
            kaggle_input / "idrid",
            work_root / "data"
        ]
        data_dir = work_root / "data"
        for cand in dataset_candidates:
            if cand.exists():
                data_dir = cand
                break
        batch_size = 32
        
    # 2. Detección de Google Colab
    elif is_colab():
        env = "colab"
        work_root = Path("/content")
        data_dir = work_root / "data"
        batch_size = 32
        
    # 3. Entorno Local (Windows / Linux / macOS)
    else:
        env = "local"
        current = Path.cwd().resolve()
        # Buscar hacia arriba hasta encontrar la carpeta 'data' o 'results'
        p = current
        found_root = None
        while p != p.parent:
            if (p / "data").exists() or (p / "results").exists():
                found_root = p
                break
            p = p.parent
        
        if found_root:
            work_root = found_root
        else:
            work_root = Path(__file__).resolve().parent.parent

        data_dir = work_root / "data"
        batch_size = 16

    # Rutas comunes derivadas
    idrid_root = data_dir / "IDRiD2" / "IDRiD" if (data_dir / "IDRiD2" / "IDRiD").exists() else data_dir / "IDRiD"
    train_images = idrid_root / "train" / "images"
    test_images = idrid_root / "test" / "images"
    train_csv = data_dir / "train_split.csv"
    val_csv = data_dir / "val_split.csv"
    output_dir = work_root / "results"

    config = {
        "env": env,
        "work_root": work_root,
        "data_dir": data_dir,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "train_images": train_images,
        "test_images": test_images,
        "output_dir": output_dir,
        "batch_size": batch_size
    }

    return config

if __name__ == "__main__":
    cfg = get_env_paths()
    print("=== Configuración de Entorno Detectada ===")
    print(f"Entorno: {cfg['env'].upper()}")
    print(f"work_root: {cfg['work_root']}")
    print(f"data_dir: {cfg['data_dir']} (Existe: {cfg['data_dir'].exists()})")
    print(f"train_csv: {cfg['train_csv']} (Existe: {cfg['train_csv'].exists()})")
    print(f"val_csv: {cfg['val_csv']} (Existe: {cfg['val_csv'].exists()})")
    print(f"train_images: {cfg['train_images']} (Existe: {cfg['train_images'].exists()})")
    print(f"test_images: {cfg['test_images']} (Existe: {cfg['test_images'].exists()})")
    print(f"batch_size recomendado: {cfg['batch_size']}")
