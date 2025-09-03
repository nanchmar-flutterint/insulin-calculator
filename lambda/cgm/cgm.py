# Use this code snippet in your app.
# If you need more information about configurations
# or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developer/language/python/
from datetime import UTC, datetime, timedelta, timezone
import boto3
from botocore.exceptions import ClientError
import json
from pydexcom import Dexcom

SECRET_NAME = "prod/cgm"
REGION_NAME = "us-east-1"

DEVICE_ID = "4714F324-F357-4D3C-9B44-660E7BB4BE04"
SESSION = boto3.session.Session()
CLIENT = SESSION.client(service_name="secretsmanager", region_name=REGION_NAME)


def lambda_handler(event, context):
    # Create a Secrets Manager client
    print(event)
    current_time = datetime.now(timezone.utc)
    formatted_date = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        get_secret_value_response = json.loads(
            CLIENT.get_secret_value(SecretId=SECRET_NAME)["SecretString"]
        )
        dexcom = Dexcom(
            username=get_secret_value_response["username"],
            password=get_secret_value_response["password"],
            region="ous",
        )  # `region="ous"
        glucose_reading = dexcom.get_current_glucose_reading()
        blood_glucose = glucose_reading.mmol_l
        trend_number = glucose_reading.trend
        trend_direction = glucose_reading.trend_direction
        trend_description = glucose_reading.trend_description
        trend_arrow = glucose_reading.trend_arrow
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1").Table("CGM")
        dynamodb.put_item(
            Item={
                "DeviceId": str(DEVICE_ID),
                "Date": str(formatted_date),
                "BloodGlucose": str(blood_glucose),
                "TrendNumber": str(trend_number),
                "TrendDirection": str(trend_direction),
                "TrendDescription": str(trend_description),
                "TrendArrow": str(trend_arrow),
                "TTL": int((datetime.now(UTC) + timedelta(days=7)).timestamp()),
            }
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e


if __name__ == "__main__":
    lambda_handler(None, None)
