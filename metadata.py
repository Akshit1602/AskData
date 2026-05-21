dim_station_column_descriptions = {
    "station_id": "Unique identifier for each Shell fuel station (e.g., BLR-001).",
    "station_name": "Display name of the station used for business reporting.",
    "city": "Metro/urban location where the station operates.",
    "cluster": "Operational grouping of stations for performance management.",
    "latitude": "Geographic latitude of the station, useful for mapping and routing.",
    "longitude": "Geographic longitude of the station.",
    "opened_year": "Year the station started operations, helpful for lifecycle and maturity analysis.",
    "has_ev_charger": "Indicates if the station supports EV charging (1 = Yes, 0 = No).",
    "cstore_size_sqft": "Size of the Shell convenience store in square feet, used for revenue potential analysis."
}

fact_station_column_descriptions = {
    "date": "Daily date for transaction reporting in YYYY-MM-DD format.",
    "station_id": "Unique identifier linking to dim_station for each Shell site.",
    "city": "City-level filter for localized performance analytics.",
    "product_family": "Fuel type sold: Petrol, Diesel, or Premium.",
    "shell_price_inr_per_liter": "Shell retail selling price per liter in INR.",
    "comp_min_price_inr_per_liter_within_3km": "Minimum competitor price detected within a 3km radius.",
    "price_gap_inr_per_liter": "Pricing difference: Shell price minus competitor minimum price (positive → Shell is pricier).",
    "liters_sold": "Total fuel volume sold for the product that day (in liters).",
    "revenue_inr": "Total revenue generated from fuel sales in INR.",
    "gross_margin_inr": "Gross margin earned on fuel sales for the day in INR.",
    "downtime_minutes": "Pump/equipment downtime duration impacting sales opportunity.",
    "stockout_flag": "Indicates if a product was unavailable for any duration (1 = Stockout, 0 = Normal).",
    "promo_active": "Indicates whether a discount/offer/promotion was active (1 = Yes, 0 = No).",
    "competitors_within_3km": "Number of competitor stations competing for the same catchment area.",
    "weather_heat_index": "Approximate temperature/humidity index that influences fuel demand.",
    "rainfall_mm": "Rainfall amount that may impact footfall and demand fluctuations.",
    "holiday_flag": "Marks national/major holidays that drive demand changes (1 = Holiday).",
    "footfall_estimate": "Estimated number of customers visiting the station on that day.",
    "cstore_transactions": "Number of completed transactions in the convenience store.",
    "cstore_revenue_inr": "Revenue generated from non-fuel C-store sales.",
    "loyalty_signups": "Number of new enrollments into Shell loyalty programmes.",
    "ev_charger_sessions": "Count of EV charging sessions (if facility exists)."
}

relationships = {
    "fact_station_day_product": {
        "station_id": {
            "references": {
                "table": "dim_station",
                "column": "station_id"
            },
            "relationship_type": "many_to_one"
        }
    }
}

table_info_combined = (
    "dim_station(station_id, station_name, city, cluster, latitude, longitude, opened_year, has_ev_charger, cstore_size_sqft)\n"
    "fact_station(date, station_id, city, product_family, shell_price_inr_per_liter, "
    "comp_min_price_inr_per_liter_within_3km, price_gap_inr_per_liter, liters_sold, revenue_inr, gross_margin_inr, "
    "downtime_minutes, stockout_flag, promo_active, competitors_within_3km, weather_heat_index, rainfall_mm, "
    "holiday_flag, footfall_estimate, cstore_transactions, cstore_revenue_inr, loyalty_signups, ev_charger_sessions)\n"
)
