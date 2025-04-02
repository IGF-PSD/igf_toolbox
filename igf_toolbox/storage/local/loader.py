# Importation des modules
# Modules de base
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from geopandas import read_file


# Fonction de chargement des données depuis un jeu de données en local
def load_local(filepath: str, **kwargs) -> Any:
    """Load data from a local file based on its extension.

    Args:
        filepath (str): Path to the local file
        **kwargs: Additional arguments for reading the file:
            - CSV: encoding, separator, etc.
            - Excel: sheet_name, skiprows, etc.
            - JSON: encoding, etc.
            - Others: format-specific options

    Returns:
        Any: The loaded data in appropriate format:
            - .xlsx, .xls -> pandas DataFrame
            - .csv -> pandas DataFrame
            - .json -> dict or pandas DataFrame
            - .pkl -> pickled object
            - .geojson -> GeoDataFrame
            - .parquet -> pandas DataFrame

    Raises:
        ValueError: If the file extension is not supported
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty

    Examples:
        Load CSV with specific options:
        >>> data = load_local(
        ...     'data.csv',
        ...     encoding='utf-8',
        ...     sep=';'
        ... )

        Load specific Excel sheet:
        >>> data = load_local(
        ...     'report.xlsx',
        ...     sheet_name='Sales',
        ...     skiprows=1
        ... )

        Load GeoJSON:
        >>> geodata = load_local('map.geojson')
    """
    # Convert string path to Path object for better handling
    path = Path(filepath)
    extension = path.suffix.lower()[1:]  # Remove the dot and convert to lowercase

    # Handle different file types
    if extension == "xlsx":
        return pd.read_excel(filepath, engine="openpyxl", **kwargs)
    elif extension == "xls":
        return pd.read_excel(filepath, engine="xlrd", **kwargs)
    elif extension == "csv":
        return pd.read_csv(filepath, **kwargs)
    elif extension == "json":
        with open(filepath, "r") as f:
            return json.load(f, **kwargs)
    elif extension == "pkl":
        return pd.read_pickle(filepath, **kwargs)
    elif extension == "geojson":
        return read_file(filepath, **kwargs)
    elif extension == "parquet":
        return pd.read_parquet(filepath, **kwargs)
    else:
        raise ValueError(
            "Invalid extension: should be in ['xlsx', 'xls', 'json', 'pkl', 'geojson', 'csv', 'parquet']."
        )
