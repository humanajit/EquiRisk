import io
import os
import boto3
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "equirisk-data-lake")


def get_s3_client():
    """Initializes and returns a boto3 S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


def upload_dataframe_to_s3(
    df: pd.DataFrame, s3_key: str, file_format: str = "parquet"
) -> bool:
    """Streams a pandas DataFrame directly into an S3 bucket without saving locally.

    :param df: Pandas DataFrame to upload
    :param s3_key: Destination path in S3 (e.g.
        'raw/yahoo_financials/ticker=RELIANCE/data.parquet')
    :param file_format: 'parquet' or 'csv'
    """
    s3_client = get_s3_client()
    buffer = io.BytesIO()

    try:
        if file_format == "parquet":
            df.to_parquet(buffer, index=False, engine="pyarrow")
        elif file_format == "csv":
            df.to_csv(buffer, index=False)
        else:
            raise ValueError("Unsupported file format. Use 'parquet' or 'csv'.")

        buffer.seek(0)
        s3_client.upload_fileobj(buffer, S3_BUCKET_NAME, s3_key)
        print(f"Successfully uploaded: s3://{S3_BUCKET_NAME}/{s3_key}")
        return True

    except Exception as e:
        print(f"Failed to upload to S3: {str(e)}")
        return False