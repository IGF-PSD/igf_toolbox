# Importation des modules
# Modules de base
from io import BytesIO
# Module de gestion du format JSON
from json import dumps
# Module de gestion du format pickle
from pickle import dump
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
import xlsxwriter
# Module de gestion des données graphiques
from matplotlib.pyplot import close, savefig

# Importation du module de connection
from ._connection import _S3Connection


# Classe de sauvegarde de données sur un Bucket S3
class S3Saver(_S3Connection):
    """
    A class for saving data to an Amazon S3 bucket using 'boto3' or 's3fs' as the underlying package.

    This class extends the `_S3Connection` parent class and provides methods for establishing a connection to the S3 bucket
    and saving data to a specified S3 object.

    Args:
        package (str): The package to use for connecting to S3 ('s3fs' or 'boto3').

    Methods:
        connect(**kwargs):
            Establishes a connection to the S3 bucket.

        save(bucket, key, obj=None, **kwargs):
            Saves an object to a specified S3 object based on its file extension and object type.

    Example :
    >>> s3_saver = S3Saver(package='boto3')
    >>> s3_connection = s3_saver.connect(aws_access_key_id='your_access_key', aws_secret_access_key='your_secret_key')
    >>> s3_saver.save(bucket='your_bucket', key='your_file.csv', obj=dataframe)
    """

    def __init__(self, package: Optional[str] = "boto3") -> None:
        """
        Initialize the S3Saver class with the specified package.

        Args:
            package (str): The package to use for connecting to S3 ('s3fs' or 'boto3').
        """
        # Initialisation du parent
        super().__init__(package=package)

    def connect(self, **kwargs) -> None:
        """
        Establish a connection to the S3 bucket.

        Args:
            **kwargs: Additional keyword arguments for establishing the connection.

        Returns:
            obj: The established S3 connection.

        Example usage:
        >>> s3_saver = S3Saver(package='boto3')
        >>> s3_connection = s3_saver.connect(aws_access_key_id='your_access_key', aws_secret_access_key='your_secret_key')
        """
        # Etablissement d'une connection
        return self._connect(**kwargs)

    def save(
        self, bucket: str, key: str, obj: Optional[Union[object, None]] = None, **kwargs
    ) -> None:
        """
        Save an object to a specified S3 object based on its file extension and object type.

        Args:
            bucket (str): The name of the S3 bucket.
            key (str): The key of the S3 object to save.
            obj (obj): The object to save (Pandas DataFrame, dictionary, Pickle object, Matplotlib figure, etc.).
            **kwargs: Additional keyword arguments for saving the object.

        Raises:
            ValueError: If the 'extension' argument is not one of ['csv', 'xlsx', 'xls', 'json', 'pkl', 'png', 'parquet].

        Example :
        >>> s3_saver = S3Saver(package='boto3')
        >>> s3_connection = s3_saver.connect(aws_access_key_id='your_access_key', aws_secret_access_key='your_secret_key')
        >>> s3_saver.save(bucket='your_bucket', key='your_file.csv', obj=dataframe)
        """
        # Etablissement d'une connexion s'il n'en existe pas une nouvelle
        if not hasattr(self, "s3"):
            self.connect()

        # Extraction de l'extension du fichier à charger
        extension = key.split(".")[-1]

        # Exportation de l'objet
        if self.package == "boto3":
            if extension == "csv":
                self.s3.put_object(Bucket=bucket, Key=key, Body=obj.to_csv(**kwargs))
            elif extension in ["xlsx", "xls"]:
                # Si l'objet est un dictionnaire de DataFrame, un jeu de données est exporté par feuille
                if isinstance(obj, dict):
                    # Construction de l'objet à exporter
                    with BytesIO() as output:
                        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                            for key_obj, value_obj in obj.items():
                                # La longueur d'une sheet_name est majoré à 31 caractères
                                export_key = (
                                    key_obj if len(key_obj) <= 31 else key_obj[:31]
                                )
                                value_obj.to_excel(
                                    writer, sheet_name=export_key, **kwargs
                                )
                        output_data = output.getvalue()
                elif isinstance(obj, pd.DataFrame):
                    with BytesIO() as output:
                        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                            obj.to_excel(writer, **kwargs)
                        output_data = output.getvalue()
                # Exportation de l'objet
                self.s3.put_object(Bucket=bucket, Key=key, Body=output_data)
            elif extension == "json":
                if isinstance(obj, pd.DataFrame):
                    self.s3.put_object(
                        Bucket=bucket, Key=key, Body=obj.to_json(**kwargs)
                    )
                else:
                    self.s3.put_object(Bucket=bucket, Key=key, Body=dumps(obj))
            elif extension == "pkl":
                with BytesIO() as output:
                    dump(obj, output)
                    output_data = output.getvalue()
                self.s3.put_object(Bucket=bucket, Key=key, Body=output_data)
            elif extension == "png":
                # Construction de l'objet à exporter
                with BytesIO() as output:
                    savefig(output, format="png", **kwargs)
                    output_data = output.getvalue()
                # Exportation de l'objet
                self.s3.put_object(Bucket=bucket, Key=key, Body=output_data)
                # Fermeture des figures
                close("all")
            elif extension == "parquet":
                # Construction de l'objet à exporter
                with BytesIO() as output:
                    obj.to_parquet(output, **kwargs)
            elif extension == "geojson":
                self.s3.put_object(
                    Bucket=bucket, Key=key, Body=obj.to_json().encode("utf-8")
                )
            else:
                raise ValueError(
                    "File type should either be csv, xlsx, xls, json, pkl, geojson or png."
                )

        elif self.package == "s3fs":
            # Distinction suivant le format du fichier et export
            if extension in ["xlsx", "xls"]:
                with self.s3.open(f"{bucket}/{key}", "wb") as s3_file:
                    # Si l'objet est un dictionnaire de DataFrame, un jeu de données est exporté par feuille
                    if isinstance(obj, dict):
                        # Construction de l'objet à exporter
                        with BytesIO() as output:
                            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                                for key_obj, value_obj in obj.items():
                                    # La longueur d'une sheet_name est majorée à 31 caractères
                                    export_key = (
                                        key_obj if len(key_obj) <= 31 else key_obj[:31]
                                    )
                                    value_obj.to_excel(
                                        writer, sheet_name=export_key, **kwargs
                                    )
                            output_data = output.getvalue()
                        # Exportation de l'objet
                        s3_file.write(output_data)
                    elif isinstance(obj, pd.DataFrame):
                        # Exportation de l'objet
                        with pd.ExcelWriter(s3_file, engine="xlsxwriter") as writer:
                            obj.to_excel(writer, **kwargs)
            elif extension == "parquet":
                with self.s3.open(f"{bucket}/{key}", "wb") as s3_file:
                    obj.to_parquet(s3_file)
            elif extension == "png":
                with self.s3.open(f"{bucket}/{key}", "wb") as s3_file:
                    # Construction de l'objet à exporter
                    with io.BytesIO() as output:
                        plt.savefig(output, format="png", **kwargs)
                        output_data = output.getvalue()
                    # Exportation de l'objet
                    s3_file.write(output_data)
                    # Fermeture des figures
                    plt.close("all")
            else:
                with self.s3.open(f"{bucket}/{key}", "w") as s3_file:
                    # Distinction suivant le format du fichier et export
                    if extension == "csv":
                        obj.to_csv(s3_file, **kwargs)
                    elif extension == "json":
                        if isinstance(obj, pd.DataFrame):
                            s3_file.write(obj.to_json(**kwargs))
                        else:
                            s3_file.write(dumps(obj))
                    elif extension == "pkl":
                        obj.to_pickle(s3_file, **kwargs)
                    elif extension == "geojson":
                        obj.to_file(s3_file, **kwargs)
                    else:
                        raise ValueError(
                            "File type should either be csv, xlsx, xls, json, pkl, parquet, geojson or png."
                        )
