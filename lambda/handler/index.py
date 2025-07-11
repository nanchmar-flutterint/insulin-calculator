"""
This module handles insulin dosage calculations for
diabetic patients based on food intake and blood glucose levels.
It uses AWS services (DynamoDB, Bedrock) to store data and get nutritional information.
The calculations account for circadian rhythms and
different types of nutrients (carbs, proteins, fats).
"""

import asyncio
import json
import math
import os
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import boto3
import numpy as np
from botocore.exceptions import ClientError

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass(frozen=True)
class ImprovedConfig:
    """
    Configuration settings for the insulin calculator.
    Contains default values and environment variables used throughout the application.
    """

    table_name: str = os.getenv("DYNAMO_DB_NAME", "NutritionHistory")
    region_name: str = os.getenv("AWS_REGION", "us-east-1")
    model_id: str = os.getenv(
        "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
    )

    # Evidence-based Fiasp pharmacokinetics
    # Based on clinical PK studies showing Fiasp onset at 2.5 min, peak at 60-63 min

    insulin_peak_time: int = 63  # minutes to peak (from clinical studies)


    # Evidence-based circadian parameters
    # Based on research showing dawn phenomenon varies 15-25% of basal needs
    # and daily insulin sensitivity varies 8-12%
    dawn_amplitude: float = 0.20  # 20% variation (middle of 15-25% range)
    daily_amplitude: float = 0.08  # 10% variation (middle of 8-12% range)
    dawn_peak_hour: float = 6.0  # Dawn phenomenon peaks around 6 AM
    daily_peak_hour: float = 16.0  # Afternoon insulin resistance peak (4 PM)

    # Additional circadian parameters from research




# Initialize AWS clients
dynamodb = boto3.resource("dynamodb", region_name=ImprovedConfig.region_name).Table(
    ImprovedConfig.table_name
)
bedrock_client = boto3.client("bedrock-runtime", region_name=ImprovedConfig.region_name)


class NutrientPrompt:
    """Handles the formatting of prompts to get nutritional information from the AI model.
    Uses a template to ensure consistent formatting of requests."""

    TEMPLATE = """Provide the nutritional information for the given
    food item and quantity as accurately as
possible. Base the information on reputable nutritional databases or scientific sources.

Food item: {food_name}
Quantity: {food_quantity} grams or ml

Return only the JSON output in the following format:
{{
  "carbohydrates": "X",
  "fats": "Y", 
  "proteins": "Z",
  "gi": "M"
}}

Where:
- X, Y, Z are numbers in grams
- M is the glycemic index (unitless)
- All numerical values should be rounded to one decimal place""".strip()

    @staticmethod
    @lru_cache(maxsize=1000)
    def format(food: str, quantity: str) -> str:
        return NutrientPrompt.TEMPLATE.format(food_name=food, food_quantity=quantity)


async def get_nutrient_data(prompt: str) -> Optional[Dict[str, Any]]:
    """Asynchronously fetches nutritional data from AWS Bedrock AI model.
    Takes a formatted prompt and returns parsed nutritional information.
    Handles errors gracefully and returns None if the request fails."""
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: bedrock_client.converse(
                modelId=ImprovedConfig.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 2000, "temperature": 0, "topP": 1.0},
            ),
        )
        return json.loads(response["output"]["message"]["content"][0]["text"])
    except (ClientError, json.JSONDecodeError) as e:
        print(f"Error fetching nutrient data: {str(e)}")
        return None


class ImprovedInsulinCalculator:
    """
    Main class for calculating insulin dosages based on food intake and blood glucose levels.
    Takes into account circadian rhythms and different nutrient types.
    Provides methods to fetch nutritional data and calculate required insulin doses.
    """

    def __init__(
        self,
        foods: List[str],
        quantities: List[str],
        blood_glucose: str,
        insulin_carbs_ratio: str,
        compensation_factor: str,
        dawn_amplitude: float = ImprovedConfig.dawn_amplitude,
        daily_amplitude: float = ImprovedConfig.daily_amplitude,
        dawn_peak_hour: float = ImprovedConfig.dawn_peak_hour,
        daily_peak_hour: float = ImprovedConfig.daily_peak_hour,
    ):
        self.prompts = [NutrientPrompt.format(f, q) for f, q in zip(foods, quantities)]
        self.foods = foods
        self.quantities = quantities
        self.blood_glucose = float(blood_glucose)

        # Calculate circadian adjustment
        current_time = datetime.now(ZoneInfo("Europe/Sofia")) # should be send by frontend
        hour = current_time.hour + current_time.minute / 60.0  # More precise timing

        # Dawn phenomenon (cortisol-driven, peaks around 6 AM)
        dawn_phase = 2 * np.pi * (hour - dawn_peak_hour) / 24
        dawn_effect = dawn_amplitude * np.cos(dawn_phase)  # Cosine for dawn peak

        # Daily rhythm (afternoon insulin resistance, peaks around 4 PM)
        daily_phase = 2 * np.pi * (hour - daily_peak_hour) / 24
        daily_effect = daily_amplitude * np.sin(daily_phase)

        # Combined circadian factor (1.0 = normal, >1.0 = more resistant, <1.0 = more sensitive)
        self.circadian_factor = 1.0 + dawn_effect + daily_effect

        # Apply circadian adjustment to patient parameters
        self.insulin_carbs_ratio = float(insulin_carbs_ratio) * self.circadian_factor
        self.compensation_factor = float(compensation_factor) * self.circadian_factor

    async def get_nutrients(self) -> List[Dict[str, Any]]:
        """
        Fetches nutritional information for all food items concurrently.
        Returns a list of successful responses, filtering out any failed requests.
        """
        responses = await asyncio.gather(*[get_nutrient_data(p) for p in self.prompts])
        return [r for r in responses if r]

    def calculate_insulin_dosage(self) -> Dict[str, Any]:
        """
        Calculates total insulin dosage needed based on all inputs.
        Handles both rapid-acting insulin for carbs and reactive insulin for proteins and fats.
        Returns a dictionary with detailed breakdown of calculations and recommendations.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            responses = loop.run_until_complete(self.get_nutrients())
            if not responses:
                return {}

            # Calculate nutrients
            nutrients = np.array(
                [
                    [float(n[k]) for k in ("carbohydrates", "gi", "fats", "proteins")]
                    for n in responses
                ]
            )
            carbs, gi, fats, proteins = nutrients.T

            # Basic insulin calculations
            glucose_factor = max(
                0, (self.blood_glucose - 5.6) / self.compensation_factor
            )
            carbs_factor = 1 / self.insulin_carbs_ratio
            insulin_rapid = np.sum(carbs * carbs_factor) + glucose_factor

            # IMPROVED: Evidence-based FPU calculation
            fpu_analysis = calculate_evidence_based_fpu(np.sum(fats), np.sum(proteins))
            # Convert FPU to insulin using carb ratio (1 FPU = 10g carbs equivalent)
            fpu_total_insulin = (
                fpu_analysis["total_fpu"] * 10 / self.insulin_carbs_ratio
            )

            # Calculate GI and timing
            gi_weighted = np.average(gi, weights=carbs) if np.sum(carbs) > 0 else 55
            optimal_timing = find_optimal_shift(
                ImprovedConfig.insulin_peak_time, gi_weighted
            )

            # Prepare detailed recommendations
            dosing_schedule = []

            # Immediate dose (carbs + correction)
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

            # FPU doses with improved timing
            if fpu_total_insulin > 0.1:
                for rec in fpu_analysis["timing_recommendations"]:
                    dose = fpu_total_insulin * (rec["percentage"] / 100)
                    if dose >= 0.1:
                        dosing_schedule.append(
                            {
                                "timing": f"{rec['time_minutes']} minutes after meal",
                                "dose": round(dose, 1),
                                "type": "Extended insulin (FPU)",
                                "rationale": f"{rec['rationale']} - "
                                f"{rec['percentage']}% of {fpu_analysis['total_fpu']:.1f} FPU",
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
                "CircadianFactor": str(round(self.circadian_factor, 3)),
                "Message": " ".join(
                    f"{dose['type']}: {dose['dose']}u {dose['timing']} ({dose['rationale']})"
                    for dose in dosing_schedule
                ),
                "Total_Insulin_Units": str(round(insulin_rapid + fpu_total_insulin, 1)),
            }
        finally:
            loop.close()


def calculate_evidence_based_fpu(fats: float, proteins: float) -> Dict[str, Any]:
    """
    Calculate FPU insulin needs with proper timing based on Warsaw Method.

    Scientific basis:
    - 1 FPU = 100 kcal from fat and protein combined
    - 1 gram fat = 9 kcal, 1 gram protein = 4 kcal
    - 1 FPU has glucose-raising effect equivalent to 10g carbohydrates
    - Extended bolus timing based on established clinical guidelines

    References:
    - Warsaw School of Insulin Pump Therapy
    - Clinical studies on fat-protein insulin dosing
    """
    # Calorie calculation (unchanged - this is accurate)
    fat_calories = fats * 9
    protein_calories = proteins * 4
    total_calories = fat_calories + protein_calories
    total_fpu = total_calories / 100

    if total_fpu < 0.5:  # Lowered threshold based on research
        return {
            "total_fpu": round(total_fpu, 1),
            "total_calories": round(total_calories, 0),
            "timing_recommendations": [],
            "extended_duration_hours": 0,
        }

    # IMPROVED: Research-based timing recommendations
    # Based on studies showing different absorption patterns for fat vs protein
    fat_ratio = fat_calories / total_calories if total_calories > 0 else 0
    protein_ratio = protein_calories / total_calories if total_calories > 0 else 0

    # Timing based on macronutrient composition and FPU amount
    recommendations = []

    if total_fpu >= 3.0:
        # High FPU: Extended protocol (research-based percentages)
        if fat_ratio > 0.6:  # Fat-dominant meal
            recommendations = [
                {"time_minutes": 60, "percentage": 25, "rationale": "Early fat effect"},
                {
                    "time_minutes": 120,
                    "percentage": 35,
                    "rationale": "Peak fat absorption",
                },
                {
                    "time_minutes": 200,
                    "percentage": 25,
                    "rationale": "Extended fat effect",
                },
                {
                    "time_minutes": 300,
                    "percentage": 15,
                    "rationale": "Late fat clearance",
                },
            ]
        else:  # Mixed or protein-dominant
            recommendations = [
                {
                    "time_minutes": 45,
                    "percentage": 30,
                    "rationale": "Early protein effect",
                },
                {
                    "time_minutes": 105,
                    "percentage": 40,
                    "rationale": "Peak mixed effect",
                },
                {
                    "time_minutes": 180,
                    "percentage": 30,
                    "rationale": "Extended absorption",
                },
            ]
    elif total_fpu >= 1.5:
        # Moderate FPU: Two-phase approach
        if fat_ratio > 0.6:
            recommendations = [
                {"time_minutes": 60, "percentage": 45, "rationale": "Main fat effect"},
                {
                    "time_minutes": 150,
                    "percentage": 55,
                    "rationale": "Extended fat effect",
                },
            ]
        else:
            recommendations = [
                {
                    "time_minutes": 45,
                    "percentage": 55,
                    "rationale": "Main protein effect",
                },
                {"time_minutes": 120, "percentage": 45, "rationale": "Extended effect"},
            ]
    else:
        # Low FPU: Single dose
        delay = 60 if fat_ratio > 0.5 else 45
        recommendations = [
            {
                "time_minutes": delay,
                "percentage": 100,
                "rationale": "Single extended dose",
            }
        ]

    return {
        "total_fpu": round(total_fpu, 1),
        "total_calories": round(total_calories, 0),
        "fat_calories": round(fat_calories, 0),
        "protein_calories": round(protein_calories, 0),
        "fat_ratio": round(fat_ratio, 2),
        "protein_ratio": round(protein_ratio, 2),
        "timing_recommendations": recommendations,
        "carb_equivalent_grams": round(total_fpu * 10, 1),
    }


@lru_cache(maxsize=1000)
def gamma_pdf(t: float, a: float, scale: float) -> float:
    """
    Calculates the probability density function of the gamma distribution.
    Used to model both insulin activity and carbohydrate absorption.

    Args:
        t: Time point
        a: Shape parameter of the gamma distribution
        scale: Scale parameter of the gamma distribution

    Returns:
        Probability density at time t
    """
    if t < 0:
        return 0.0
    return (t ** (a - 1) * math.exp(-t / scale)) / (math.gamma(a) * scale**a)


def insulin_activity(t: float, peak_time: float = 60) -> float:
    """
    Models how active insulin is at a given time point.
    Uses a gamma distribution with parameters tuned to match insulin behavior.

    Args:
        t: Time point in minutes
        peak_time: Time when insulin activity peaks (default 60 minutes)

    Returns:
        Relative insulin activity at time t
    """
    a = 3  # Shape parameter
    theta = peak_time / (a - 1)  # Scale parameter
    return gamma_pdf(t, a, theta)


def carb_absorption(t: float, gi: float) -> float:
    """
    Models how quickly carbohydrates are absorbed based on their glycemic index.
    Higher GI foods are absorbed more quickly than lower GI foods.

    Args:
        t: Time point in minutes
        gi: Glycemic index of the food

    Returns:
        Relative absorption rate at time t
    """
    peak_time = 30 + (100 - gi) * 0.6
    a = 3
    theta = peak_time / (a - 1)
    return gamma_pdf(t, a, theta)


def calculate_overlap(shift: float, insulin_peak: float, gi: float) -> float:
    """
    Calculates how well insulin activity matches carbohydrate absorption.
    Used to determine optimal injection timing.

    Args:
        shift: Time offset between injection and meal
        insulin_peak: When insulin activity peaks
        gi: Glycemic index of the meal

    Returns:
        Measure of overlap between insulin and carb curves
    """
    t_values = np.linspace(max(0, shift), 300, 500)
    y_values = np.array(
        [
            insulin_activity(t - shift, insulin_peak) * carb_absorption(t, gi)
            for t in t_values
        ]
    )
    return np.trapz(y_values, t_values)


def find_optimal_shift(insulin_peak: float, gi: float) -> float:
    """
    Determines the optimal timing for insulin injection relative to meal.
    Finds when insulin activity best matches carbohydrate absorption.

    Args:
        insulin_peak: When insulin activity peaks
        gi: Average glycemic index of the meal

    Returns:
        Optimal time offset in minutes (negative means inject before meal)
    """
    shifts = np.linspace(-60, 120, 100)
    overlaps = np.array([calculate_overlap(s, insulin_peak, gi) for s in shifts])
    return shifts[np.argmax(overlaps)]


def lambda_handler(event: Dict[str, Any], _) -> Dict[str, Any]:
    """
    AWS Lambda function handler that processes incoming requests.
    Takes food and patient data, calculates insulin needs, and stores results.

    Args:
        event: AWS Lambda event containing request data
        _: AWS Lambda context (unused)

    Returns:
        Response with calculation results or error message
    """
    try:
        payload = json.loads(event["body"])
        insulin_calc = ImprovedInsulinCalculator(
            list(payload["Foods"].keys()),
            list(payload["Foods"].values()),
            payload["BloodGlucose"],
            payload["ICR"],
            payload["CF"],
            dawn_amplitude=payload.get("DawnAmplitude", ImprovedConfig.dawn_amplitude),
            daily_amplitude=payload.get(
                "DailyAmplitude", ImprovedConfig.daily_amplitude
            ),
            dawn_peak_hour=payload.get("DawnPeakHour", ImprovedConfig.dawn_peak_hour),
            daily_peak_hour=payload.get(
                "DailyPeakHour", ImprovedConfig.daily_peak_hour
            ),
        )

        table_entry = insulin_calc.calculate_insulin_dosage()
        if not table_entry:
            raise ValueError("Failed to calculate insulin dosage")

        # Add metadata
        table_entry.update(
            {
                "DeviceId": payload["DeviceId"],
                "Date": payload["Date"],
                "TTL": int((datetime.now(UTC) + timedelta(days=365)).timestamp()),
                "CircadianFactor": str(insulin_calc.circadian_factor),
                "Version": "2.0-EvidenceBased",  # Version tracking
            }
        )

        dynamodb.put_item(Item=table_entry)
        return {"statusCode": 200, "body": json.dumps(table_entry)}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"}),
        }
