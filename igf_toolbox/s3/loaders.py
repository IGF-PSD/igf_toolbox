# Importation des modules
# Module de base
from io import BytesIO
# Modules de gestion de formats JSON, Excel et de données géographiques
from json import load
from typing import Optional

import openpyxl
import pandas as pd
from geopandas import read_file

# Importation du module de connection
from ._connection import _S3Connection


class S3Loader(_S3Connection):
    """
    A class for loading data from an Amazon S3 bucket using 'boto3' or 's3fs' as the underlying package.

    This class extends the `_S3Connection` parent class and provides methods for establishing a connection to the S3 bucket
    and loading data from a specified S3 object.

    Args:
        package (str): The package to use for connecting to S3 ('s3fs' or 'boto3').

    Methods:
        connect(**kwargs):
            Establishes a connection to the S3 bucket.

        load(bucket, key, **kwargs):
            Loads data from a specified S3 object based on its file extension and returns it as a Pandas DataFrame, JSON object,
            Pickle object, or GeoDataFrame, depending on the file extension.

    Example :
    >>> s3_loader = S3Loader(package='boto3')
    >>> s3_connection = s3_loader.connect(aws_access_key_id='your_access_key', aws_secret_access_key='your_secret_key')
    >>> data = s3_loader.load(bucket='your_bucket', key='your_file.csv')
    """

    def __init__(self, package: Optional[str] = "boto3") -> None:
        """
        Initialize the S3Loader class with the specified package.

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

        Example :
        >>> s3_loader = S3Loader(package='boto3')
        >>> s3_connection = s3_loader.connect(aws_access_key_id='your_access_key', aws_secret_access_key='your_secret_key')
        """
        # Etablissement d'une connection
        return self._connect(**kwargs)

    def load(self, bucket: str, key: str, **kwargs) -> None:
        """
        Load data from a specified S3 object based on its file extension.

        Args:
            bucket (str): The name of the S3 bucket.
            key (str): The key of the S3 object to load.
            **kwargs: Additional keyword arguments for reading the data.

        Returns:
            obj: The loaded data (Pandas DataFrame, JSON object, Pickle object, or GeoDataFrame).

        Example :
        >>> s3_loader = S3Loader(package='boto3')
        >>> s3_connection = s3_loader.connect(aws_access_key_id='your_access_key', aws_secret_access_key='your_secret_key')
        >>> data = s3_loader.load(bucket='your_bucket', key='your_file.csv')
        """
        # Etablissement d'une connexion s'il n'en existe pas une nouvelle
        if not hasattr(self, "s3"):
            self.connect()

        # Extraction de l'extension du fichier à charger
        extension = key.split(".")[-1]

        # Chargement des données
        if self.package == "boto3":
            # Ouverture du fichier
            s3_file = self.s3.get_object(Bucket=bucket, Key=key)["Body"]
            # Test suivant l'extension du fichier à charger et lecture de ce-dernier
            if extension == "xlsx":
                data = pd.read_excel(s3_file.read(), engine="openpyxl", **kwargs)
            elif extension == "xls":
                data = pd.read_excel(s3_file.read(), engine="xlrd", **kwargs)
            elif extension == "parquet":
                data = pd.read_parquet(BytesIO(s3_file.read()), **kwargs)
            else:
                data = self._read_data(s3_file=s3_file, extension=extension, **kwargs)
        elif self.package == "s3fs":
            with self.s3.open(f"{bucket}/{key}", "rb") as s3_file:
                # Test suivant l'extension du fichier à charger et lecture de ce-dernier
                if extension == "xlsx":
                    data = pd.read_excel(s3_file, engine="openpyxl", **kwargs)
                elif extension == "xls":
                    data = pd.read_excel(s3_file, engine="xlrd", **kwargs)
                else:
                    data = self._read_data(
                        s3_file=s3_file, extension=extension, **kwargs
                    )

        return data

    # Fonction auxiliaire de lecture des données
    def _read_data(self, s3_file, extension: str, **kwargs):
        """
        Read data from an S3 file based on its extension.

        This function reads data from an S3 file based on its file extension and returns the data as a Pandas DataFrame,
        JSON object, Pickle object, or GeoDataFrame, depending on the extension.

        Args:
            s3_file (obj): The S3 file object to read from.
            extension (str): The file extension indicating the file format ('csv', 'json', 'pkl', 'geojson', 'parquet').
            **kwargs: Additional keyword arguments specific to the file format's reading method.

        Returns:
            obj: The read data (Pandas DataFrame, JSON object, Pickle object, or GeoDataFrame).

        Raises:
            ValueError: If the 'extension' argument is not one of ['csv', 'json', 'pkl', 'geojson', 'parquet'].

        Example :
        >>> s3_connection = _S3Connection(package='boto3')
        >>> s3_client = s3_connection._connect(aws_access_key_id='your_access_key', aws_secret_access_key='your_secret_key')
        >>> s3_file = s3_client.get_object(Bucket='your_bucket', Key='your_file.csv')
        >>> data = _read_data(s3_file, extension='csv')
        """
        # Test sur l'extension et lecture du fichier
        if extension == "csv":
            data = pd.read_csv(s3_file, **kwargs)
        elif extension == "json":
            data = json.load(s3_file, **kwargs)
        elif extension == "pkl":
            data = pd.read_pickle(s3_file, **kwargs)
        elif extension == "geojson":
            data = read_file(s3_file, **kwargs)
        elif extension == "parquet":
            data = pd.read_parquet(s3_file, **kwargs)
        else:
            raise ValueError(
                "Invalid extension : should be in ['xlsx', 'xls', 'json', 'pkl', 'geojson', 'csv', 'parquet']."
            )
        return data
