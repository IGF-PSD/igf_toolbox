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
    """A class for saving data to Amazon S3 buckets.

    This class extends the `_S3Connection` parent class and provides methods for
    establishing connections to S3 buckets and saving data to S3 objects.

    Args:
        s3_package (str, optional): Package to use for S3 connections ('s3fs' or 'boto3').
            Defaults to "boto3".

    Attributes:
        s3: The S3 connection object (initialized when needed)
        s3_package (str): The package being used for S3 connectivity

    Examples:
        Save DataFrame to S3 using boto3:
        >>> saver = S3Saver()
        >>> saver.connect(
        ...     aws_access_key_id='YOUR_KEY',
        ...     aws_secret_access_key='YOUR_SECRET'
        ... )
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> saver.save(
        ...     bucket='my-bucket',
        ...     key='path/to/file.csv',
        ...     obj=df
        ... )

        Save multiple DataFrames to Excel using s3fs:
        >>> saver = S3Saver(s3_package='s3fs')
        >>> saver.connect()  # Uses environment variables
        >>> sheets = {'Sheet1': df1, 'Sheet2': df2}
        >>> saver.save(
        ...     bucket='my-bucket',
        ...     key='path/to/file.xlsx',
        ...     obj=sheets
        ... )
    """

    def __init__(self, s3_package: Optional[str] = "boto3") -> None:
        """Initialize the S3Saver with specified S3 package.

        Args:
            s3_package (str, optional): Package to use for S3 connections.
                Must be either 's3fs' or 'boto3'. Defaults to "boto3".
        """
        # Initialisation du parent
        super().__init__(s3_package=s3_package)

    def connect(self, **kwargs) -> None:
        """
        Establish a connection to the S3 bucket.

        Args:
            **kwargs: Additional keyword arguments for establishing the connection.

        Returns:
            object: The established S3 connection.

        Example usage:
        >>> s3_saver = S3Saver(package='boto3')
        >>> s3_connection = s3_saver.connect(
            aws_access_key_id='your_access_key',
            aws_secret_access_key='your_secret_key'
        )
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
            obj (object): The object to save (DataFrame, dictionary, etc.).
            **kwargs: Additional keyword arguments for saving the object.

        Raises:
            ValueError: If the file extension is not supported
            TypeError: If the object type doesn't match the requirements

        Example :
        >>> s3_saver = S3Saver(package='boto3')
        >>> s3_connection = s3_saver.connect(
            aws_access_key_id='your_access_key',
            aws_secret_access_key='your_secret_key'
        )
        >>> s3_saver.save(bucket='your_bucket', key='your_file.csv', obj=dataframe)
        """
        # Etablissement d'une connexion s'il n'en existe pas une nouvelle
        if not hasattr(self, "s3"):
            self.connect()

        # Extraction de l'extension du fichier à charger
        extension = key.split(".")[-1]

        # Exportation de l'objet
        if self.s3_package == "boto3":
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

        elif self.s3_package == "s3fs":
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
                    with BytesIO() as output:
                        savefig(output, format="png", **kwargs)
                        output_data = output.getvalue()
                    # Exportation de l'objet
                    s3_file.write(output_data)
                    # Fermeture des figures
                    close("all")
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
