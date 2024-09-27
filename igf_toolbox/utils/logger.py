# Importation des modules
# Modules de base
import logging
import os


# Fonction d'initialisation du logger
def _init_logger(filename: os.PathLike) -> logging.Logger:
    """
    Initializes the logger for logging to a file.

    Parameters:
    - filename (os.PathLike): Path to the log file.

    Returns:
    - logging.Logger: Initialized logger object.

    Note:
    - This function configures logging to output messages to both console and a file.
    """
    # Configuration de logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8",
        level=logging.INFO,
    )

    # Vérification de l'existance du dossier pour le fichier de log
    log_directory = os.path.dirname(filename)

    if (not os.path.exists(log_directory)) & (log_directory != ""):
        os.makedirs(log_directory)

    # Configuration du fichier de logs
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)

    # Initialisation du logger
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    return logger
