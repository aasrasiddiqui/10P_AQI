import os
import tempfile
import hopsworks
from dotenv import load_dotenv

load_dotenv()

if os.name == "nt":
    TEMP_DIR = r"C:\tmp"

    os.makedirs(
        TEMP_DIR,
        exist_ok=True
    )

    os.environ["TEMP"] = TEMP_DIR
    os.environ["TMP"] = TEMP_DIR
    os.environ["TMPDIR"] = TEMP_DIR

    tempfile.tempdir = TEMP_DIR


def main():
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name="karachi_aqi_features",
        version=2
    )

    print("Reading feature group...")

    df = fg.read()

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())


if __name__ == "__main__":
    main()