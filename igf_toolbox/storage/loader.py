# Importation des modules
from typing import Any, Optional

# Module de chargement de fichiers en local
from .local.loader import load_local
# Module de chargement de fichiers depuis S3
from .s3.loader import S3Loader


# Classe générale de chargement des données
class Loader(S3Loader):
    """A unified class for loading data from both S3 and local storage.

    This class provides functionality to load data from either Amazon S3 or local storage
    based on whether a bucket name is specified. It inherits from S3Loader to handle
    S3-specific operations.

    Args:
        s3_package (str, optional): The package to use for S3 connections ('s3fs' or 'boto3').
            Defaults to "boto3".

    Attributes:
        s3: The S3 connection object, initialized when needed.

    Examples:
        Load data from S3:
        >>> loader = Loader(s3_package='boto3')
        >>> s3_data = loader.load(
        ...     filepath='data/sales.csv',
        ...     bucket='my-bucket',
        ...     aws_access_key_id='YOUR_KEY',
        ...     aws_secret_access_key='YOUR_SECRET'
        ... )

        Load data from local storage:
        >>> loader = Loader()
        >>> local_data = loader.load(filepath='data/sales.csv')

    Notes:
        - When loading from S3, AWS credentials can be provided either through
          environment variables or as parameters.
        - For local loading, all standard formats are supported: CSV, Excel, JSON,
          Pickle, GeoJSON, and Parquet.
    """

    def __init__(self, s3_package: Optional[str] = "boto3") -> None:
        """Initialize the Loader with specified S3 package.

        Args:
            s3_package (str, optional): Package to use for S3 connections.
                Must be either 's3fs' or 'boto3'. Defaults to "boto3".
        """
        super().__init__(s3_package=s3_package)

    def load(self, filepath: str, bucket: Optional[str] = None, **kwargs) -> Any:
        """Load data from either S3 or local storage.

        Args:
            filepath (str): Path to the file. For S3, this is the key within the bucket.
                For local storage, this is the path on the filesystem.
            bucket (str, optional): S3 bucket name. If None, loads from local storage.
            **kwargs: Additional arguments passed to the underlying loader:
                - For S3: aws_access_key_id, aws_secret_access_key, aws_session_token,
                  endpoint_url, verify
                - For both: file format specific options (encoding, separator, etc.)

        Returns:
            Any: The loaded data in appropriate format based on file extension:
                - .csv, .xlsx, .xls -> pandas DataFrame
                - .json -> dict or pandas DataFrame
                - .pkl -> pickled object
                - .geojson -> GeoDataFrame
                - .parquet -> pandas DataFrame

        Raises:
            ValueError: If the file extension is not supported
            FileNotFoundError: If the local file doesn't exist
            botocore.exceptions.ClientError: If there are S3 access issues

        Examples:
            Load CSV from S3:
            >>> data = loader.load(
            ...     filepath='data.csv',
            ...     bucket='my-bucket',
            ...     aws_access_key_id='KEY',
            ...     aws_secret_access_key='SECRET'
            ... )

            Load local Excel file with specific options:
            >>> data = loader.load(
            ...     filepath='data.xlsx',
            ...     sheet_name='Sales',
            ...     skiprows=1
            ... )
        """
        if bucket is not None:
            # Extract S3-specific kwargs
            s3_kwargs = {
                k: kwargs.pop(k)
                for k in [
                    "aws_access_key_id",
                    "aws_secret_access_key",
                    "aws_session_token",
                    "endpoint_url",
                    "verify",
                ]
                if k in kwargs
            }
            # Connect if needed
            if not hasattr(self, "s3"):
                self.connect(**s3_kwargs)
            # Use parent S3Loader's load method
            return super().load(bucket=bucket, key=filepath, **kwargs)
        else:
            # Use parent LocalLoader's load_local method
            return load_local(filepath=filepath, **kwargs)
