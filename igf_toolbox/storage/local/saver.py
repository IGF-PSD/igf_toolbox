# Importation des modules
# Modules de base
import json
from pathlib import Path
from pickle import dump
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


# Fonction de sauvegarde de données en local
def save_local(filepath: str, obj: Optional[object] = None, **kwargs) -> None:
    """Save an object to a local file based on its extension.

    Args:
        filepath (str): Path where to save the file
        obj: The object to save. Type requirements depend on the file extension:
            - .csv, .parquet: pandas DataFrame
            - .xlsx, .xls: pandas DataFrame or dict of DataFrames
            - .json: Any JSON-serializable object or pandas DataFrame
            - .pkl: Any picklable object
            - .png: Active matplotlib figure
            - .geojson: GeoDataFrame
        **kwargs: Additional arguments for saving:
            - CSV: index, encoding, etc.
            - Excel: sheet_name, index, etc.
            - JSON: indent, orient, etc.
            - Others: format-specific options

    Raises:
        ValueError: If the file extension is not supported
        TypeError: If obj type doesn't match the file extension requirements
        IOError: If there are issues writing to the file
        OSError: If the directory structure doesn't exist

    Examples:
        Save DataFrame to CSV:
        >>> save_local(
        ...     'data.csv',
        ...     df,
        ...     index=False,
        ...     encoding='utf-8'
        ... )

        Save multiple DataFrames to Excel sheets:
        >>> sheets = {
        ...     'Sales': sales_df,
        ...     'Costs': costs_df
        ... }
        >>> save_local('report.xlsx', sheets)

        Save matplotlib figure:
        >>> plt.plot([1, 2, 3])
        >>> save_local('plot.png', dpi=300)

    Notes:
        - Parent directories will be created if they don't exist
        - For Excel files with multiple sheets, sheet names are truncated to 31 chars
    """
    # Convert string path to Path object and ensure parent directory exists
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get file extension
    extension = path.suffix.lower()[1:]

    # Handle different file types
    if extension == "csv":
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Object must be a pandas DataFrame for CSV export")
        obj.to_csv(path, **kwargs)

    elif extension in ["xlsx", "xls"]:
        if isinstance(obj, dict):
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                for key_obj, value_obj in obj.items():
                    # Excel sheet names limited to 31 characters
                    export_key = key_obj if len(key_obj) <= 31 else key_obj[:31]
                    value_obj.to_excel(writer, sheet_name=export_key, **kwargs)
        elif isinstance(obj, pd.DataFrame):
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                obj.to_excel(writer, **kwargs)
        else:
            raise TypeError(
                "Object must be a DataFrame or dict of DataFrames for Excel export"
            )

    elif extension == "json":
        if isinstance(obj, pd.DataFrame):
            obj.to_json(path, **kwargs)
        else:
            with open(path, "w") as f:
                json.dump(obj, f, **kwargs)

    elif extension == "pkl":
        with open(path, "wb") as f:
            dump(obj, f, **kwargs)

    elif extension == "png":
        plt.savefig(path, format="png", **kwargs)
        plt.close("all")

    elif extension == "parquet":
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Object must be a pandas DataFrame for Parquet export")
        obj.to_parquet(path, **kwargs)

    elif extension == "geojson":
        if not isinstance(obj, gpd.GeoDataFrame):
            raise TypeError("Object must be a GeoDataFrame for GeoJSON export")
        obj.to_file(path, driver="GeoJSON", **kwargs)

    else:
        raise ValueError(
            "File type should either be csv, xlsx, xls, json, pkl, parquet, geojson or png."
        )
