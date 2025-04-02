# Importation des modules
# Module de base
# Modules de gestion de formats JSON, Excel et de données géographiques
import json
from io import BytesIO
from typing import Optional

import openpyxl
import pandas as pd
from geopandas import read_file

# Importation du module de connection
from ._connection import _S3Connection


class S3Loader(_S3Connection):
    """A class for loading data from Amazon S3 buckets.

    This class extends the `_S3Connection` parent class and provides methods for
    establishing connections to S3 buckets and loading data from S3 objects.

    Args:
        s3_package (str, optional): Package to use for S3 connections ('s3fs' or 'boto3').
            Defaults to "boto3".

    Attributes:
        s3: The S3 connection object (initialized when needed)
        s3_package (str): The package being used for S3 connectivity

    Examples:
        Load CSV from S3 using boto3:
        >>> loader = S3Loader()
        >>> loader.connect(
        ...     aws_access_key_id='YOUR_KEY',
        ...     aws_secret_access_key='YOUR_SECRET'
        ... )
        >>> data = loader.load(
        ...     bucket='my-bucket',
        ...     key='path/to/file.csv'
        ... )

        Load Excel file using s3fs:
        >>> loader = S3Loader(s3_package='s3fs')
        >>> loader.connect()  # Uses environment variables
        >>> data = loader.load(
        ...     bucket='my-bucket',
        ...     key='path/to/file.xlsx',
        ...     sheet_name='Data'
        ... )
    """

    def __init__(self, s3_package: Optional[str] = "boto3") -> None:
        """Initialize the S3Loader with specified S3 package.

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
            object: The loaded data (Pandas DataFrame, JSON object, Pickle object, or GeoDataFrame).

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
        if self.s3_package == "boto3":
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
        elif self.s3_package == "s3fs":
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
        """Read data from an S3 file based on its extension.

        Internal method to handle reading of data from S3 files based on their format.

        Args:
            s3_file: The S3 file object to read from (type varies by s3_package)
            extension (str): File extension indicating format
            **kwargs: Additional arguments passed to the reading function

        Returns:
            object: The loaded data in appropriate format:
                - .csv -> pandas DataFrame
                - .json -> dict or pandas DataFrame
                - .pkl -> pickled object
                - .geojson -> GeoDataFrame
                - .parquet -> pandas DataFrame

        Raises:
            ValueError: If the extension is not supported
            pd.errors.EmptyDataError: If the file is empty
            json.JSONDecodeError: If JSON file is invalid

        Example:
            >>> s3_file = s3.get_object(Bucket='bucket', Key='file.csv')['Body']
            >>> data = loader._read_data(s3_file, 'csv', encoding='utf-8')
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
