import os

METADATA = {
    "shell": {
        "domain_context": "expert data analyst for Shell Retail (India)",
        "files": [
            {
                "path": "Shell__dim_station__preview_.csv",
                "table": "dim_station",
                "format": "csv"
            },
            {
                "path": "Shell__fact_station_day_product__preview_.csv",
                "table": "fact_station",
                "format": "csv",
                "date_col": "date",
                "date_format": "%d-%m-%Y"
            }
        ],
        "table_descriptions": {
            "dim_station": "Contains master data about Shell fuel stations including location, facilities, and opening details.",
            "fact_station": "Contains daily transactional and operational data for each station and product family."
        },
        "column_descriptions": {
            "dim_station": {
                "station_id": "Unique identifier for each Shell fuel station (e.g., BLR-001).",
                "station_name": "Display name of the station used for business reporting.",
                "city": "Metro/urban location where the station operates.",
                "cluster": "Operational grouping of stations for performance management.",
                "latitude": "Geographic latitude of the station, useful for mapping and routing.",
                "longitude": "Geographic longitude of the station.",
                "opened_year": "Year the station started operations, helpful for lifecycle and maturity analysis.",
                "has_ev_charger": "Indicates if the station supports EV charging (1 = Yes, 0 = No).",
                "cstore_size_sqft": "Size of the Shell convenience store in square feet, used for revenue potential analysis."
            },
            "fact_station": {
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
        },
        "relationships": {
            "fact_station": {
                "station_id": {
                    "references": {
                        "table": "dim_station",
                        "column": "station_id"
                    },
                    "relationship_type": "many_to_one"
                }
            }
        },
        "table_info_combined": (
            "dim_station(station_id, station_name, city, cluster, latitude, longitude, opened_year, has_ev_charger, cstore_size_sqft)\n"
            "fact_station(date, station_id, city, product_family, shell_price_inr_per_liter, "
            "comp_min_price_inr_per_liter_within_3km, price_gap_inr_per_liter, liters_sold, revenue_inr, gross_margin_inr, "
            "downtime_minutes, stockout_flag, promo_active, competitors_within_3km, weather_heat_index, rainfall_mm, "
            "holiday_flag, footfall_estimate, cstore_transactions, cstore_revenue_inr, loyalty_signups, ev_charger_sessions)\n"
        )
    },
    "sample": {
        "domain_context": "expert business analyst for retail campaign performance, analyzing customer behavior across different product hierarchy levels (L1, L0, Walmart)",
        "files": [
            {
                "path": "Sample Dataset.xlsx",
                "table": "campaign_data",
                "format": "excel",
                "sheet_name": "Data"
            }
        ],
        "table_descriptions": {
            "campaign_data": "Contains retail campaign performance metrics, customer counts, and sales values across different organizational levels."
        },
        "column_descriptions": {
            "campaign_data": {
                "group_name": "The experimental group (e.g., Treatment or Control).",
                "cohort": "The customer segment (e.g., Acquisition or Retention).",
                "frequency": "Customer shopping frequency category.",
                "HH_CNT": "Total Household count.",
                "L1_HH_CNT": "Household count at Level 1 (SKU hierarchy).",
                "L1_HH_GMV": "Gross Merchandise Value at Level 1.",
                "L0_HH_CNT": "Household count at Level 0 (Category hierarchy).",
                "L0_HH_GMV": "Gross Merchandise Value at Level 0.",
                "WMT_HH_CNT": "Household count at Walmart level (Total store hierarchy).",
                "WMT_HH_GMV": "Gross Merchandise Value at Walmart level.",
                "New_L1": "Count of new customers at Level 1.",
                "Repeat_L1": "Count of repeat customers at Level 1.",
                "Reactivated_L1": "Count of reactivated customers at Level 1.",
                "Orders_L1": "Total number of orders at Level 1.",
                "Quantity_L1": "Total quantity of items sold at Level 1.",
                "New_L0": "Count of new customers at Level 0.",
                "Repeat_L0": "Count of repeat customers at Level 0.",
                "Reactivated_L0": "Count of reactivated customers at Level 0.",
                "Orders_L0": "Total number of orders at Level 0.",
                "Quantity_L0": "Total quantity of items sold at Level 0.",
                "New_WMT": "Count of new customers at Walmart level.",
                "Repeat_WMT": "Count of repeat customers at Walmart level.",
                "Reactivated_WMT": "Count of reactivated customers at Walmart level.",
                "Orders_WMT": "Total number of orders at Walmart level.",
                "Quantity_WMT": "Total quantity of items sold at Walmart level."
            }
        },
        "relationships": {},
        "table_info_combined": (
            "campaign_data(group_name, cohort, frequency, HH_CNT, L1_HH_CNT, L1_HH_GMV, L0_HH_CNT, L0_HH_GMV, WMT_HH_CNT, WMT_HH_GMV, New_L1, Repeat_L1, Reactivated_L1, Orders_L1, Quantity_L1, New_L0, Repeat_L0, Reactivated_L0, Orders_L0, Quantity_L0, New_WMT, Repeat_WMT, Reactivated_WMT, Orders_WMT, Quantity_WMT)\n"
        )
    }
}

def get_active_dataset_name():
    return os.getenv("ACTIVE_DATASET", "shell")

def get_metadata():
    dataset_name = get_active_dataset_name()
    return METADATA.get(dataset_name, METADATA["shell"])
