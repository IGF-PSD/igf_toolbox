# Importation des modules
# Modules de base
from typing import Optional

from .local.saver import save_local
# Module ad hoc
from .s3.saver import S3Saver


class Saver(S3Saver):
    """A unified class for saving data to both S3 and local storage.

    This class provides functionality to save data to either Amazon S3 or local storage
    based on whether a bucket name is specified. It inherits from S3Saver to handle
    S3-specific operations.

    Args:
        s3_package (str, optional): The package to use for S3 connections ('s3fs' or 'boto3').
            Defaults to "boto3".

    Attributes:
        s3: The S3 connection object, initialized when needed.

    Examples:
        Save DataFrame to S3:
        >>> saver = Saver(s3_package='boto3')
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> saver.save(
        ...     filepath='data/output.csv',
        ...     bucket='my-bucket',
        ...     obj=df,
        ...     aws_access_key_id='YOUR_KEY',
        ...     aws_secret_access_key='YOUR_SECRET'
        ... )

        Save DataFrame locally:
        >>> saver = Saver()
        >>> saver.save(filepath='data/output.csv', obj=df)

        Save multiple DataFrames to Excel sheets:
        >>> sheets = {
        ...     'Sheet1': df1,
        ...     'Sheet2': df2
        ... }
        >>> saver.save(filepath='output.xlsx', obj=sheets)
    """

    def __init__(self, s3_package: Optional[str] = "boto3"):
        """Initialize the Saver with specified S3 package.

        Args:
            s3_package (str, optional): Package to use for S3 connections.
                Must be either 's3fs' or 'boto3'. Defaults to "boto3".
        """
        super().__init__(s3_package=s3_package)

    def save(
        self,
        filepath: str,
        obj: Optional[object] = None,
        bucket: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Save data to either S3 or local storage.

        Args:
            filepath (str): Path for saving the file. For S3, this is the key within
                the bucket. For local storage, this is the path on the filesystem.
            obj: The object to save. Type requirements depend on the file extension:
                - .csv, .parquet: pandas DataFrame
                - .xlsx, .xls: pandas DataFrame or dict of DataFrames
                - .json: Any JSON-serializable object or pandas DataFrame
                - .pkl: Any picklable object
                - .png: Active matplotlib figure
                - .geojson: GeoDataFrame
            bucket (str, optional): S3 bucket name. If None, saves to local storage.
            **kwargs: Additional arguments for saving:
                - For S3: aws_access_key_id, aws_secret_access_key, aws_session_token,
                  endpoint_url, verify
                - For both: format-specific options (index, encoding, etc.)

        Raises:
            ValueError: If the file extension is not supported
            TypeError: If obj type doesn't match the file extension requirements
            IOError: If there are issues writing to local storage
            botocore.exceptions.ClientError: If there are S3 access issues

        Examples:
            Save DataFrame to CSV in S3:
            >>> saver.save(
            ...     filepath='data.csv',
            ...     bucket='my-bucket',
            ...     obj=df,
            ...     index=False
            ... )

            Save DataFrame to local Excel with specific options:
            >>> saver.save(
            ...     filepath='data.xlsx',
            ...     obj=df,
            ...     sheet_name='Data',
            ...     index=False
            ... )

            Save multiple DataFrames to Excel sheets:
            >>> sheets = {'Sales': sales_df, 'Costs': costs_df}
            >>> saver.save(
            ...     filepath='report.xlsx',
            ...     obj=sheets
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
            # Use parent S3Saver's save method
            super().save(bucket=bucket, key=filepath, obj=obj, **kwargs)
        else:
            # Use LocalSaver's save_local method
            save_local(filepath=filepath, obj=obj, **kwargs)
