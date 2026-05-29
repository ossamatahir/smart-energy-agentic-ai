
import random
from datetime import datetime

class WeatherTool:
    name = "WeatherTool"

    def run(self):
        conditions = ["Sunny", "Partly Cloudy", "Overcast", "Cloudy"]
        condition  = random.choice(conditions)
        temp       = round(random.uniform(28, 42), 1)
        cloud      = {"Sunny": 5, "Partly Cloudy": 30,
                      "Overcast": 70, "Cloudy": 85}[condition]
        return {
            "location"   : "Karachi, Pakistan",
            "condition"  : condition,
            "temperature": temp,
            "cloud_cover": cloud,
        }
