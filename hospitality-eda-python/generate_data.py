"""
generate_data.py
Generates a simulated hotel booking dataset (city hotel + resort hotel) for EDA.

Why simulated? Real hotel data is confidential; a simulated set lets us control
realistic relationships (seasonality, lead time vs. cancellation, ADR by room
type) and deliberately inject missing values and outliers so every EDA
technique can be demonstrated. A fixed seed makes results reproducible.
"""
import numpy as np
import pandas as pd

SEED = 42
N = 6000
rng = np.random.default_rng(SEED)


def generate(n: int = N) -> pd.DataFrame:
    hotel = rng.choice(["City Hotel", "Resort Hotel"], n, p=[0.62, 0.38])

    # Arrival dates across 2023-2024 with summer / December peaks
    days = pd.date_range("2023-01-01", "2024-12-31")
    doy = days.dayofyear.values
    w = 1 + 0.6 * np.exp(-((doy - 200) ** 2) / (2 * 40 ** 2)) + 0.3 * (doy > 350)
    arrival = rng.choice(days, n, p=w / w.sum())
    arrival = pd.to_datetime(arrival)

    segment = rng.choice(
        ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"],
        n, p=[0.45, 0.17, 0.18, 0.12, 0.08])

    # Lead time (days): groups & offline book earlier
    base_lead = rng.gamma(2.0, 35, n)
    base_lead *= np.where(segment == "Groups", 1.8, 1)
    base_lead *= np.where(segment == "Offline TA/TO", 1.3, 1)
    base_lead *= np.where(segment == "Corporate", 0.5, 1)
    lead_time = np.clip(base_lead.round(), 0, 450).astype(int)

    weekend_nights = rng.choice([0, 1, 2, 3], n, p=[0.40, 0.35, 0.20, 0.05])
    week_nights = rng.choice([0, 1, 2, 3, 4, 5, 6], n,
                             p=[0.03, 0.20, 0.27, 0.22, 0.13, 0.10, 0.05])
    week_nights = np.where((weekend_nights == 0) & (week_nights == 0), 1, week_nights)
    stay_nights = weekend_nights + week_nights

    adults = rng.choice([1, 2, 3, 4], n, p=[0.17, 0.66, 0.12, 0.05])
    children = rng.choice([0, 1, 2], n, p=[0.88, 0.08, 0.04])

    room_type = rng.choice(["Standard", "Deluxe", "Suite", "Family"], n,
                           p=[0.50, 0.30, 0.08, 0.12])

    # ADR: room type, hotel, seasonality, noise
    room_base = pd.Series(room_type).map(
        {"Standard": 85, "Deluxe": 120, "Suite": 210, "Family": 140}).values
    hotel_mult = np.where(hotel == "Resort Hotel", 1.12, 1.0)
    season = 1 + 0.35 * np.exp(-((arrival.dayofyear.values - 205) ** 2) / (2 * 35 ** 2))
    adr = room_base * hotel_mult * season * rng.normal(1, 0.10, n)
    adr += 12 * children + 6 * (adults - 2)
    adr = np.round(np.clip(adr, 30, None), 2)

    special_requests = rng.choice([0, 1, 2, 3], n, p=[0.55, 0.28, 0.12, 0.05])
    prev_cancel = rng.choice([0, 1, 2], n, p=[0.93, 0.05, 0.02])
    parking = rng.choice([0, 1], n, p=[0.88, 0.12])

    # Cancellation probability driven by lead time, segment, history, requests
    logit = (-1.9 + 0.006 * lead_time
             + np.where(segment == "Online TA", 0.5, 0)
             + np.where(segment == "Groups", 0.4, 0)
             + np.where(segment == "Corporate", -0.6, 0)
             + 1.2 * (prev_cancel > 0)
             - 0.45 * special_requests - 0.5 * parking)
    p_cancel = 1 / (1 + np.exp(-logit))
    is_canceled = (rng.random(n) < p_cancel).astype(int)

    country = rng.choice(["PRT", "GBR", "FRA", "DEU", "ESP", "USA", "IND", "Other"],
                         n, p=[0.28, 0.14, 0.12, 0.10, 0.09, 0.07, 0.05, 0.15])

    df = pd.DataFrame({
        "booking_id": np.arange(1, n + 1),
        "hotel": hotel,
        "arrival_date": arrival,
        "lead_time": lead_time,
        "stay_nights": stay_nights,
        "weekend_nights": weekend_nights,
        "week_nights": week_nights,
        "adults": adults,
        "children": children,
        "market_segment": segment,
        "room_type": room_type,
        "country": country,
        "adr": adr,
        "special_requests": special_requests,
        "previous_cancellations": prev_cancel,
        "parking_spaces": parking,
        "is_canceled": is_canceled,
    })

    # ---- Inject data-quality issues (so cleaning / outlier steps are meaningful)
    out_idx = rng.choice(n, 25, replace=False)
    df.loc[out_idx, "adr"] = rng.uniform(450, 900, 25).round(2)        # extreme rates
    df.loc[rng.choice(n, 15, replace=False), "lead_time"] = rng.integers(500, 700, 15)
    df.loc[rng.choice(n, 120, replace=False), "country"] = np.nan      # missing
    df.loc[rng.choice(n, 60, replace=False), "children"] = np.nan      # missing
    df = pd.concat([df, df.sample(20, random_state=SEED)], ignore_index=True)  # duplicates
    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv("data/hotel_bookings.csv", index=False)
    print(f"Saved data/hotel_bookings.csv  shape={data.shape}")
