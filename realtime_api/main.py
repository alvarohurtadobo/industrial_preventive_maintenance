from fastapi import FastAPI
from datetime import datetime
import zoneinfo
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

timezone_countries = {
    "US": "America/New_York",
    "CA": "America/Toronto",
    "MX": "America/Mexico_City",
    "GB": "Europe/London",
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "IT": "Europe/Rome",
    "ES": "Europe/Madrid",
    "NL": "Europe/Amsterdam",
}

@app.get("/time/{iso_code}")
async def time(iso_code: str):
    iso = iso_code.upper()
    timezone_string = timezone_countries.get(iso)
    if not timezone_string:
        return {"error": "Invalid ISO code"}
    tz = zoneinfo.ZoneInfo(timezone_string)
    return {"time": datetime.now(tz)}