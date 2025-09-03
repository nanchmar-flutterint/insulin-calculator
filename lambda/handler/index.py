import asyncio
import json
import math
import os
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional

import boto3
import numpy as np
from botocore.exceptions import ClientError

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass(frozen=True)
class Config:
    table_name: str = os.getenv("DYNAMO_DB_NAME", "NutritionHistory")
    region_name: str = os.getenv("AWS_REGION", "us-east-1")
    model_id: str = os.getenv(
        "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    insulin_peak_time: int = 63
    SYSTEM_PROMPT = """You provide precise nutritional data using USDA FoodData Central or similar reputable sources. 
    Calculate values for the exact quantity specified, not per 100g.
    Response Format - JSON only:
    {
     "carbohydrates": "X.X",
     "fats": "Y.Y", 
     "proteins": "Z.Z",
     "calories": "W.W",
     "gi": "M.M"
    }

    Rules:
    - All values in grams/calories for specified quantity
    - Round to one decimal place
    - GI: 0-100 scale, use closest similar food if unavailable
    - If food not found: {"error": "Food item not found or insufficient data available"}
    - Use closest equivalent if exact food unavailable

    Calculation: (base_per_100g × quantity) ÷ 100, rounded to 1 decimal."""


dynamodb = boto3.resource("dynamodb", region_name=Config.region_name).Table(
    Config.table_name
)
bedrock_client = boto3.client("bedrock-runtime", region_name=Config.region_name)


class NutrientPrompt:
    TEMPLATE = """
    Provide the nutritional information for the given food item and quantity as accurately as possible. 
    Base the information 
    on reputable nutritional databases such as USDA FoodData Central, 
    NCCDB (Nutrition Coordinating Center Food & Nutrient Database), or peer-reviewed nutritional research.
    
    Food item: {food_name}
    Quantity: {food_quantity} grams or ml
    
    If the exact food item cannot be found, use the closest equivalent food item. 
    If glycemic index (GI) data is unavailable for the specific food, use the GI value from the closest 
    similar food item.
    If the food item cannot be identified at all, return an error message in JSON format.
    Return only the JSON output in the following format:
    
    For successful lookup: 
    {{ 
    "carbohydrates": "X",
    "fats": "Y", 
    "proteins": "Z", 
    "calories": "W", 
    "gi": "M" 
    }}
    
    For errors: 
    {{ 
    "error": "Food item not found or insufficient data available" 
    }}
    
    Where:
    X, Y, Z are numbers in grams per the specified quantity
    W is calories per the specified quantity
    M is the glycemic index (unitless, 0-100 scale)
    All numerical values should be rounded to one decimal place
    Values should be calculated for exactly {food_quantity} grams/ml of {food_name}, not per 100g
    """.strip()

    @staticmethod
    @lru_cache(maxsize=1000)
    def format(food: str, quantity: str) -> str:
        return NutrientPrompt.TEMPLATE.format(food_name=food, food_quantity=quantity)


async def get_nutrient_data(prompt: str) -> Optional[Dict[str, Any]]:
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: bedrock_client.converse(
                modelId=Config.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 300, "temperature": 0, "topP": 0.1},
                system=[{"text": Config.SYSTEM_PROMPT}],
            ),
        )
        return json.loads(response["output"]["message"]["content"][0]["text"])
    except (ClientError, json.JSONDecodeError) as e:
        print(f"Error fetching nutrient data: {str(e)}")
        return None


class InsulinCalculator:
    def __init__(
        self,
        foods: List[str],
        quantities: List[str],
        blood_glucose: str,
        insulin_carbs_ratio: str,
        compensation_factor: str,
    ):
        self.prompts = [NutrientPrompt.format(f, q) for f, q in zip(foods, quantities)]
        self.foods = foods
        self.quantities = quantities
        self.blood_glucose = float(blood_glucose)
        self.insulin_carbs_ratio = float(insulin_carbs_ratio)
        self.compensation_factor = float(compensation_factor)

    async def get_nutrients(self) -> List[Dict[str, Any]]:
        responses = await asyncio.gather(*[get_nutrient_data(p) for p in self.prompts])
        return [r for r in responses if r]

    def calculate_insulin_dosage(self) -> Dict[str, Any]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            responses = loop.run_until_complete(self.get_nutrients())
            if not responses:
                return {}
            nutrients = np.array(
                [
                    [float(n[k]) for k in ("carbohydrates", "gi", "fats", "proteins")]
                    for n in responses
                ]
            )
            carbs, gi, fats, proteins = nutrients.T
            glucose_factor = max(
                0, (self.blood_glucose - 5.6) / self.compensation_factor
            )
            carbs_factor = 1 / self.insulin_carbs_ratio
            insulin_rapid = np.sum(carbs * carbs_factor) + glucose_factor
            fpu_analysis = calculate_fpu(np.sum(fats), np.sum(proteins))
            fpu_total_insulin = (
                fpu_analysis["total_fpu"] * 10 / self.insulin_carbs_ratio
            ) * fpu_analysis["adjustment_factor"]

            gi_weighted = np.average(gi, weights=carbs) if np.sum(carbs) > 0 else 55
            optimal_timing = find_optimal_shift_with_bg(
                Config.insulin_peak_time, gi_weighted, self.blood_glucose
            )
            dosing_schedule = []
            if insulin_rapid > 0:
                dosing_schedule.append(
                    {
                        "timing": f"{abs(optimal_timing):.0f} minutes "
                        f"{'before' if optimal_timing < 0 else 'after'} meal",
                        "dose": round(insulin_rapid, 1),
                        "type": "Rapid-acting (carbs + correction)",
                        "rationale": f"Covers {np.sum(carbs):.1f}g carbs and BG correction",
                    }
                )

            if fpu_total_insulin >= 0.5:
                extended_timing = fpu_analysis["extended_duration_hours"]
                if extended_timing > 0:
                    extended_delay = 60 if extended_timing <= 2.0 else 90
                    dosing_schedule.append(
                        {
                            "timing": f"{extended_delay} minutes after meal",
                            "dose": round(fpu_total_insulin, 1),
                            "type": "Extended (fat/protein)",
                            "rationale": f"Covers {fpu_analysis['total_fpu']} FPU with {fpu_analysis['adjustment_factor']}x factor",
                        }
                    )
            return {
                "BloodGlucose": str(self.blood_glucose),
                "Carbohydrates": str(np.sum(carbs)),
                "Proteins": str(np.sum(proteins)),
                "Fats": str(np.sum(fats)),
                "InsulinRapid": str(round(insulin_rapid, 1)),
                "InsulinReactive": str(round(fpu_total_insulin, 1)),
                "ICR": str(round(self.insulin_carbs_ratio, 1)),
                "CF": str(round(self.compensation_factor, 1)),
                "GlucoseIndex": str(round(gi_weighted, 0)),
                "Foods": ",".join(self.foods),
                "Quantity": ",".join(self.quantities),
                "FPU": str(fpu_analysis["total_fpu"]),
                "Message": " ".join(
                    f"{dose['type']}: {dose['dose']}u {dose['timing']} ({dose['rationale']})"
                    for dose in dosing_schedule
                ),
                "Total_Insulin_Units": str(round(insulin_rapid + fpu_total_insulin, 1)),
            }
        finally:
            loop.close()


def calculate_fpu(fats: float, proteins: float) -> Dict[str, Any]:
    fat_calories = fats * 9
    protein_calories = proteins * 4 * 0.58
    total_calories = fat_calories + protein_calories
    total_fpu = total_calories / 100

    if total_fpu < 1.0:
        adjustment_factor = 1.0
        extended_duration_hours = 0.0
    elif total_fpu < 2.0:
        adjustment_factor = 1.15
        extended_duration_hours = 2.0
    elif total_fpu < 3.0:
        adjustment_factor = 1.25
        extended_duration_hours = 3.0
    else:
        adjustment_factor = 1.3
        extended_duration_hours = 4.0
    return {
        "total_fpu": round(total_fpu, 1),
        "total_calories": round(total_calories, 0),
        "fat_calories": round(fat_calories, 0),
        "protein_calories": round(protein_calories, 0),
        "adjustment_factor": adjustment_factor,
        "extended_duration_hours": extended_duration_hours,
    }


@lru_cache(maxsize=1000)
def gamma_pdf(t: float, a: float, scale: float) -> float:

    if t < 0:
        return 0.0
    return (t ** (a - 1) * math.exp(-t / scale)) / (math.gamma(a) * scale**a)


def insulin_activity(t: float, peak_time: float = 60) -> float:
    a = 3
    theta = peak_time / (a - 1)
    return gamma_pdf(t, a, theta)


def carb_absorption(t: float, gi: float) -> float:
    peak_time = 30 + (100 - gi) * 0.6
    a = 3
    theta = peak_time / (a - 1)
    return gamma_pdf(t, a, theta)


def calculate_overlap(shift: float, insulin_peak: float, gi: float) -> float:
    t_values = np.linspace(0, 300, 500)
    overlaps = []
    for t in t_values:
        if shift >= 0:
            ins = insulin_activity(t - shift, insulin_peak) if t >= shift else 0
        else:
            ins = insulin_activity(t - shift, insulin_peak)
        carb = carb_absorption(t, gi)
        overlaps.append(ins * carb)
    return np.trapz(overlaps, t_values)


def find_optimal_shift(insulin_peak: float, gi: float) -> float:
    shifts = np.linspace(-20, 30, 100)
    overlaps = np.array([calculate_overlap(s, insulin_peak, gi) for s in shifts])
    return shifts[np.argmax(overlaps)]


def find_optimal_shift_with_bg(
    insulin_peak: float, gi: float, blood_glucose: float
) -> float:
    base_shift = find_optimal_shift(insulin_peak, gi)
    target_bg = 5.6
    adjustment = -(blood_glucose - target_bg) * 3
    adjusted_shift = base_shift + adjustment
    if blood_glucose < 3.9:
        adjusted_shift = max(adjusted_shift, 20)
    elif blood_glucose > 13.9:
        adjusted_shift = min(adjusted_shift, 0)
    return np.clip(adjusted_shift, -20, 30)


def lambda_handler(event: Dict[str, Any], _) -> Dict[str, Any]:
    try:
        payload = json.loads(event["body"])
        insulin_calc = InsulinCalculator(
            list(payload["Foods"].keys()),
            list(payload["Foods"].values()),
            payload["BloodGlucose"],
            payload["ICR"],
            payload["CF"],
        )
        table_entry = insulin_calc.calculate_insulin_dosage()
        if not table_entry:
            raise ValueError("Failed to calculate insulin dosage")

        table_entry.update(
            {
                "DeviceId": payload["DeviceId"],
                "Date": payload["Date"],
                "TTL": int((datetime.now(UTC) + timedelta(days=7)).timestamp()),
                "Version": "2.0-EvidenceBased",
            }
        )
        dynamodb.put_item(Item=table_entry)
        device_id = payload["DeviceId"]
        dt = datetime.fromisoformat(payload["Date"].replace("Z", "+00:00"))
        safe_date = dt.strftime("%Y%m%dT%H%M%SZ")

        create_one_time_schedule("CGMLambdaFunction", 1, f"{device_id}-{safe_date}-1")
        create_one_time_schedule("CGMLambdaFunction", 2, f"{device_id}-{safe_date}-2")
        create_one_time_schedule("CGMLambdaFunction", 3, f"{device_id}-{safe_date}-3")
        create_one_time_schedule(
            "CGMLambdaFunction",
            0,
            f"{device_id}-{safe_date}-30",
            delay_minutes=30,
        )
        create_one_time_schedule(
            "CGMLambdaFunction",
            0,
            f"{device_id}-{safe_date}-15",
            delay_minutes=15,
        )
        create_one_time_schedule(
            "CGMLambdaFunction",
            0,
            f"{device_id}-{safe_date}-2",
            delay_minutes=2,
        )
        return {"statusCode": 200, "body": json.dumps(table_entry)}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"}),
        }


def create_one_time_schedule(function_name, delay_hours, rule_name, delay_minutes=0):
    scheduler = boto3.client("scheduler")
    region = os.environ["AWS_REGION"]
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]

    target_time = datetime.now(timezone.utc) + timedelta(
        hours=delay_hours, minutes=delay_minutes
    )

    response = scheduler.create_schedule(
        Name=rule_name,
        ScheduleExpression=f"at({target_time.strftime('%Y-%m-%dT%H:%M:%S')})",
        FlexibleTimeWindow={"Mode": "OFF"},
        Target={
            "Arn": f"arn:aws:lambda:{region}:{account_id}:function:{function_name}",
            "RoleArn": f"arn:aws:iam::{account_id}:role/EventBridgeSchedulerRole",
            "RetryPolicy": {"MaximumRetryAttempts": 5, "MaximumEventAgeInSeconds": 180},
        },
    )

    return response
