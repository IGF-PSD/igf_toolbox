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
    """Base class for managing connections to Amazon S3 buckets.

    This class provides the foundational functionality for connecting to S3 buckets
    using either 's3fs' or 'boto3' as the underlying package.

    Args:
        s3_package (str, optional): Package to use for S3 connections ('s3fs' or 'boto3').
            Defaults to "boto3".

    Attributes:
        s3_package (str): The package being used for S3 connectivity
        s3: The S3 connection object (initialized when needed)

    Raises:
        ValueError: If s3_package is not 's3fs' or 'boto3'

    Examples:
        Using boto3:
        >>> conn = _S3Connection()
        >>> conn._connect(
        ...     aws_access_key_id='YOUR_KEY',
        ...     aws_secret_access_key='YOUR_SECRET'
        ... )

        Using s3fs with custom endpoint:
        >>> conn = _S3Connection(s3_package='s3fs')
        >>> conn._connect(endpoint_url='https://custom.endpoint')

    Notes:
        - AWS credentials can be provided either through environment variables or
          as parameters during connection.
        - Environment variables used:
            - AWS_S3_ENDPOINT
            - AWS_ACCESS_KEY_ID
            - AWS_SECRET_ACCESS_KEY
            - AWS_SESSION_TOKEN
    """

    def __init__(self, s3_package: Optional[str] = "boto3") -> None:
        """Initialize the S3 connection with specified S3 package.

        Args:
            s3_package (str, optional): Package to use for S3 connections.
                Must be either 's3fs' or 'boto3'. Defaults to "boto3".
        """
        # Initialisation du package utilisé pour se connecter au bucket S3
        # Deux valeurs sont valides pour ce paramètre 's3fs' et 'boto3'
        if s3_package not in ["s3fs", "boto3"]:
            raise ValueError("'s3_package' must be in ['s3fs', 'boto3']")

        self.s3_package = s3_package

    def _connect(
        self,
        endpoint_url: Optional[Union[str, None]] = None,
        aws_access_key_id: Optional[Union[str, None]] = None,
        aws_secret_access_key: Optional[Union[str, None]] = None,
        aws_session_token: Optional[Union[str, None]] = None,
        verify: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """Establish connection to S3 using specified credentials.

        Args:
            endpoint_url (str, optional): Custom S3 endpoint URL
            aws_access_key_id (str, optional): AWS access key
            aws_secret_access_key (str, optional): AWS secret key
            aws_session_token (str, optional): AWS session token
            verify (bool, optional): Whether to verify SSL certificates
            **kwargs: Additional arguments passed to boto3.client or S3FileSystem

        Returns:
            _S3Connection: Self, with initialized s3 attribute

        Raises:
            ValueError: If s3_package is invalid
            botocore.exceptions.ClientError: If connection fails

        Notes:
            If credentials are not provided, they will be read from environment
            variables.
        """
        # Désactive les warnings en raison de la non vérification du certificat (non recommandé)
        disable_warnings()

        if self.s3_package == "boto3":
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
        elif self.s3_package == "s3fs":
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
            raise ValueError("'s3_package' must be in ['s3fs', 'boto3']")

        return self
