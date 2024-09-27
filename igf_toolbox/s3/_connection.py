# Importation des modules
# Modules de base
import os
from typing import Optional, Union

# Modules S3
from boto3 import client
from s3fs import S3FileSystem
# Modules de communisation avec s3
from urllib3 import disable_warnings


# Classe parent gérant la connection au bucket pour les loaders et savers
class _S3Connection:
    """
    A parent class for managing connections to an Amazon S3 bucket using different packages.

    This class allows you to connect to an S3 bucket using either 's3fs' or 'boto3' as the underlying package.
    's3fs' is used for file system-like access, while 'boto3' provides a more comprehensive AWS SDK.

    Args:
        package (str): The package to use for connecting to S3 ('s3fs' or 'boto3').

    Raises:
        ValueError: If the 'package' argument is not one of ['s3fs', 'boto3'].

    Attributes:
        package (str): The package used for S3 connectivity ('s3fs' or 'boto3').

    Methods:
        _connect(endpoint_url, aws_access_key_id, aws_secret_access_key, aws_session_token, verify, **kwargs):
            Connects to the S3 bucket with the specified parameters and package.
            Returns the connected S3 client or file system.

    Example :
    >>> s3_connection = _S3Connection(package='boto3')
    >>> s3_client = s3_connection._connect(aws_access_key_id='your_access_key', aws_secret_access_key='your_secret_key')

    Note:
        Make sure to set the required AWS environment variables for successful connections.
    """

    def __init__(self, package: Optional[str] = "boto3") -> None:
        """
        Initialize the S3 connection class with the specified package.

        Args:
            package (str): The package to use for connecting to S3 ('s3fs' or 'boto3').
        """
        # Initialisation du package utilisé pour se connecter au bucket S3
        # Deux valeurs sont valides pour ce paramètre 's3fs' et 'boto3'
        if package not in ["s3fs", "boto3"]:
            raise ValueError("'package' must be in ['s3fs', 'boto3']")

        self.package = package

    def _connect(
        self,
        endpoint_url: Optional[Union[str, None]] = None,
        aws_access_key_id: Optional[Union[str, None]] = None,
        aws_secret_access_key: Optional[Union[str, None]] = None,
        aws_session_token: Optional[Union[str, None]] = None,
        verify: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        Connects to the S3 bucket using the specified parameters and package.

        Args:
            endpoint_url (str): The S3 endpoint URL (optional).
            aws_access_key_id (str): The AWS access key ID (optional).
            aws_secret_access_key (str): The AWS secret access key (optional).
            aws_session_token (str): The AWS session token (optional).
            verify (bool): Whether to verify SSL certificates (default is False).
            **kwargs: Additional keyword arguments specific to the chosen package.

        Returns:
            obj: The connected S3 client or file system.

        Raises:
            ValueError: If the 'package' argument is not one of ['s3fs', 'boto3'].

        Example :
        >>> s3_connection = _S3Connection(package='boto3')
        >>> s3_client = s3_connection._connect(aws_access_key_id='your_access_key', aws_secret_access_key='your_secret_key')
        """
        # Désactive les warnings en raison de la non vérification du certificat (non recommandé)
        disable_warnings()

        if self.package == "boto3":
            self.s3 = client(
                "s3",
                endpoint_url=(
                    endpoint_url
                    if endpoint_url is not None
                    else "https://" + os.environ["AWS_S3_ENDPOINT"]
                ),
                aws_access_key_id=(
                    aws_access_key_id
                    if aws_access_key_id is not None
                    else os.environ["AWS_ACCESS_KEY_ID"]
                ),
                aws_secret_access_key=(
                    aws_secret_access_key
                    if aws_secret_access_key is not None
                    else os.environ["AWS_SECRET_ACCESS_KEY"]
                ),
                aws_session_token=(
                    aws_session_token
                    if aws_session_token is not None
                    else os.environ["AWS_SESSION_TOKEN"]
                ),
                verify=verify,
                **kwargs,
            )
        elif self.package == "s3fs":
            self.s3 = S3FileSystem(
                client_kwargs={
                    "endpoint_url": (
                        endpoint_url
                        if endpoint_url is not None
                        else "https://" + os.environ["AWS_S3_ENDPOINT"]
                    )
                },
                **kwargs,
            )
        else:
            raise ValueError("'package' must be in ['s3fs', 'boto3']")

        return self
