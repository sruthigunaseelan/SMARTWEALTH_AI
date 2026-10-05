import os
import random
import numpy as np
import pandas as pd

# Set fixed random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 12 Target Categories
CATEGORIES = [
    "Groceries",
    "Dining & Restaurants",
    "Utilities & Bills",
    "Subscriptions & Streaming",
    "Transportation & Fuel",
    "Healthcare & Pharmacy",
    "Entertainment & Leisure",
    "Shopping & Electronics",
    "Travel & Lodging",
    "Education & Courses",
    "Personal Care",
    "Investments & Wealth",
]

# Patterns for merchant descriptions per category
MERCHANT_PATTERNS = {
    "Groceries": [
        "WALMART SUPERCENTER #{store} {city}",
        "TARGET GROCERY STORE #{store}",
        "WHOLE FOODS MARKET {city}",
        "TRADER JOES STORE #{store}",
        "KROGER GROCERY STORE #{store}",
        "ALDI FOODS {city}",
        "SAFEWAY SUPERMARKET #{store}",
        "COSTCO WHOLESALE #{store} GROCERY",
        "FRESH DIRECT ONLINE GROCERY",
        "LOCAL CORNER GROCERY STORE",
    ],
    "Dining & Restaurants": [
        "STARBUCKS COFFEE #{store} {city}",
        "MCDONALDS RESTAURANT #{store}",
        "CHIPOTLE MEXICAN GRILL {city}",
        "SUBWAY SANDWICH SHOP #{store}",
        "DOMINOS PIZZA DELIVERY {store}",
        "OLIVE GARDEN ITALIAN RESTAURANT",
        "SWEETGREEN SALAD BAR #{store}",
        "PANERA BREAD #{store} {city}",
        "TACO BELL DRIVE THRU #{store}",
        "THE LOCAL BISTRO & CAFE",
    ],
    "Utilities & Bills": [
        "CITY POWER & LIGHT ELEC BILL",
        "METRO WATER & SEWER UTILITY",
        "PACIFIC GAS AND ELECTRIC CO",
        "COMCAST XFINITY INTERNET BILL",
        "VERIZON WIRELESS MONTHLY BILL",
        "AT&T MOBILITY CELLULAR BILL",
        "WASTE MANAGEMENT TRASH SERVICE",
        "NATIONAL GRID NATURAL GAS",
        "SPECTRUM CABLE INTERNET BILL",
        "MUNICIPAL HEAT & UTILITIES",
    ],
    "Subscriptions & Streaming": [
        "NETFLIX DIGITAL MONTHLY RECURRING",
        "SPOTIFY MUSIC PREMIUM SUBSCRIPTION",
        "AMAZON PRIME ANNUAL MEMBERSHIP",
        "HULU STREAMING SERVICE MONTHLY",
        "DISNEY PLUS DIGITAL SUBSCRIPTION",
        "APPLE ICLOUD STORAGE MONTHLY",
        "HBO MAX STREAMING MONTHLY",
        "NEW YORK TIMES DIGITAL SUBSCRIPTION",
        "YOUTUBE PREMIUM MONTHLY RECURRING",
        "CHATGPT PLUS OPENAI SUBSCRIPTION",
    ],
    "Transportation & Fuel": [
        "SHELL OIL #{store} GAS STATION",
        "CHEVRON FUEL STOP #{store}",
        "EXXONMOBIL PETROLEUM {city}",
        "BP GAS & CONVENIENCE STORE",
        "UBER RIDE TRIP SAN FRANCISCO",
        "LYFT RIDE HAIL METRO AREA",
        "METRO TRANSIT PASS RENEWAL",
        "AMTRAK RAILWAY TICKET PURCHASE",
        "PARKING METER METRO {city}",
        "EZ PASS TOLL ROAD AUTO CHARGE",
    ],
    "Healthcare & Pharmacy": [
        "CVS PHARMACY #{store} {city}",
        "WALGREENS DRUGSTORE #{store}",
        "QUEST DIAGNOSTICS LAB TEST",
        "DENTAL CARE ASSOCIATES CHECKUP",
        "VISION CARE OPTOMETRY CLINIC",
        "CITY GENERAL HOSPITAL COPAY",
        "EXPRESS SCRIPTS PRESCRIPTION",
        "MINUTECLINIC MEDICAL VISIT",
        "RITE AID PHARMACY STORE #{store}",
        "PHYSICAL THERAPY CLINIC SESSION",
    ],
    "Entertainment & Leisure": [
        "AMC THEATRES MOVIE TICKETS #{store}",
        "REGAL CINEMAS TICKET OFFICE",
        "TICKETMASTER CONCERT EVENT",
        "STEAM GAMES DIGITAL PURCHASE",
        "PLAYSTATION NETWORK DIGITAL",
        "BOWLING ALLEY & ARCADE {city}",
        "CITY ZOO & BOTANICAL GARDEN",
        "GOLF CLUB GREEN FEES",
        "EVENTBRITE TICKET ENTRY",
        "NATIONAL PARK ENTRANCE PASS",
    ],
    "Shopping & Electronics": [
        "AMAZON MARKETPLACE RETAIL ORDER",
        "BEST BUY ELECTRONICS #{store}",
        "APPLE STORE ONLINE ORDER",
        "NIKE RETAIL OUTLET STORE",
        "ZARA CLOTHING STORE {city}",
        "HOME DEPOT HARDWARE #{store}",
        "LOWES HOME IMPROVEMENT",
        "MACY DEPT STORE CLOTHING",
        "NORDSTROM RETAIL PURCHASE",
        "EBAY ONLINE MERCHANT ORDER",
    ],
    "Travel & Lodging": [
        "DELTA AIR LINES TICKET ATLANTA",
        "UNITED AIRLINES BOOKING ONLINE",
        "AMERICAN AIRLINES FLIGHT TICKET",
        "MARRIOTT HOTEL & RESORT STAY",
        "HILTON INN LODGING {city}",
        "AIRBNB VACATION RENTAL BOOKING",
        "EXPEDIA TRAVEL PACKAGE DEPOSIT",
        "HERTZ RENT A CAR LOCATION",
        "ENTERPRISE CAR RENTAL {city}",
        "BOOKING COM HOTEL RESERVATION",
    ],
    "Education & Courses": [
        "COURSERA PLUS ANNUAL SUBSCRIPTION",
        "UDEMY ONLINE COURSE PURCHASE",
        "CHEGG TEXTBOOK RENTAL MONTHLY",
        "UNIVERSITY TUITION FEE DEPOSIT",
        "DUOLINGO PLUS LANGUAGE APP",
        "CAMPUS BOOKSTORE TEXTBOOKS",
        "SKILLSHARE ONLINE ANNUAL CLASS",
        "MASTERCLASS LEARNING SUBSCRIPTION",
        "EDX ONLINE EDUCATION COURSE",
        "CODECADEMY PRO SUBSCRIPTION",
    ],
    "Personal Care": [
        "SEPHORA BEAUTY STORE #{store}",
        "ULTA BEAUTY & COSMETICS",
        "GREAT CLIPS HAIRCUT SALON",
        "SUPERCUTS BARBER SHOP #{store}",
        "PLANET FITNESS MONTHLY GYM",
        "EQUINOX FITNESS CLUB DUES",
        "SPA & MASSAGE WELLNESS CENTER",
        "BATH AND BODY WORKS STORE",
        "SALON HAIR COLOR & STYLING",
        "NAIL BAR & PEDICURE SALON",
    ],
    "Investments & Wealth": [
        "FIDELITY BROKERAGE INVEST DEPOSIT",
        "VANGUARD INDEX FUND PURCHASE",
        "CHARLES SCHWAB BROKERAGE ACCOUNT",
        "ROBINHOOD FINANCIAL DEPOSIT",
        "BETTERMENT AUTOMATED INVESTMENT",
        "WEALTHFRONT ADVISORY DEPOSIT",
        "COINBASE CRYPTO EXCHANGE DEPOSIT",
        "ACORNS AUTOMATIC ROUND UP INVEST",
        "E*TRADE SECURITIES INVESTMENT",
        "TREASURYDIRECT SAVINGS BOND",
    ],
}

CITIES = ["NEW YORK", "LOS ANGELES", "CHICAGO", "HOUSTON", "PHOENIX", "SEATTLE", "AUSTIN", "BOSTON", "DENVER", "ATLANTA"]

def generate_transaction_classification_dataset(samples_per_category=200):
    """Generates a dataset of raw transaction descriptions for category classification."""
    data = []
    for cat in CATEGORIES:
        patterns = MERCHANT_PATTERNS[cat]
        for _ in range(samples_per_category):
            pattern = random.choice(patterns)
            store_num = random.randint(100, 9999)
            city_name = random.choice(CITIES)
            tx_desc = pattern.format(store=store_num, city=city_name)
            
            # Add random noise/variations like transaction prefixes or IDs
            prefix = random.choice(["", "POS ", "POS DEBIT ", "ACH WITHDRAWAL ", "TST* ", "SQ *", "PAYPAL *", "VISA DRAFT "])
            suffix = random.choice(["", f" REF#{random.randint(10000,99999)}", f" #{random.randint(10,99)}", " ON-LINE"])
            
            raw_text = f"{prefix}{tx_desc}{suffix}".strip()
            amount = round(random.uniform(3.50, 450.00), 2)
            
            data.append({
                "raw_description": raw_text,
                "category": cat,
                "amount": amount
            })
            
    df = pd.DataFrame(data)
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df

def generate_monthly_expenses_dataset(num_users=10, months=36):
    """Generates historical monthly total expenses for time-series forecasting."""
    dates = pd.date_range(start="2023-01-01", periods=months, freq="MS")
    records = []
    
    for user_id in range(1, num_users + 1):
        base_expense = random.uniform(1800.0, 3500.0)
        trend = random.uniform(10.0, 25.0)  # gradual monthly inflation/lifestyle increase
        
        for idx, date_val in enumerate(dates):
            month_num = date_val.month
            # Seasonality: Dec high (holidays), Jul high (vacation), Jan low
            season_factor = 1.0
            if month_num == 12:
                season_factor = 1.25
            elif month_num == 7:
                season_factor = 1.15
            elif month_num == 1:
                season_factor = 0.90
            elif month_num == 11:
                season_factor = 1.10
                
            noise = random.gauss(0, 120.0)
            total_expense = round(max(500.0, (base_expense + (idx * trend)) * season_factor + noise), 2)
            
            records.append({
                "user_id": f"USER_{user_id:03d}",
                "ds": date_val.strftime("%Y-%m-%d"),
                "total_expense": total_expense,
                "month": month_num,
                "year": date_val.year
            })
            
    df = pd.DataFrame(records)
    return df

def generate_anomalous_transactions_dataset(n_samples=500):
    """Generates benchmark dataset for Isolation Forest anomaly detection."""
    records = []
    
    # 85% normal transactions, 15% anomalies
    n_normal = int(n_samples * 0.85)
    n_anom = n_samples - n_normal
    
    for i in range(n_normal):
        cat = random.choice(CATEGORIES)
        amount = round(random.gauss(45.0, 15.0), 2)
        amount = max(3.0, amount)
        price_change_pct = round(random.gauss(0.0, 0.02), 4) # normal change near 0%
        frequency_score = round(random.uniform(0.5, 1.0), 2)  # common merchant
        
        records.append({
            "tx_id": f"TX_NORM_{i:04d}",
            "raw_description": f"NORMAL MERCHANT {cat.upper()}",
            "category": cat,
            "amount": amount,
            "price_change_pct": price_change_pct,
            "merchant_rarity_score": round(1.0 - frequency_score, 2),
            "is_anomaly": 0,
            "anomaly_type": "Normal"
        })
        
    anomaly_types = ["Price Spike", "Suspicious Amount", "Rare/Unusual Merchant", "Duplicate Subscription Jump"]
    for i in range(n_anom):
        anom_type = random.choice(anomaly_types)
        cat = random.choice(CATEGORIES)
        
        if anom_type == "Price Spike":
            amount = round(random.uniform(150.0, 400.0), 2)
            price_change_pct = round(random.uniform(0.75, 3.50), 4) # 75% to 350% jump
            merchant_rarity = round(random.uniform(0.1, 0.4), 2)
        elif anom_type == "Suspicious Amount":
            amount = round(random.uniform(1500.0, 8500.0), 2) # unusually large purchase
            price_change_pct = round(random.uniform(0.0, 0.50), 4)
            merchant_rarity = round(random.uniform(0.3, 0.8), 2)
        elif anom_type == "Rare/Unusual Merchant":
            amount = round(random.uniform(80.0, 600.0), 2)
            price_change_pct = round(random.uniform(0.1, 0.40), 4)
            merchant_rarity = round(random.uniform(0.85, 0.99), 2) # rare merchant
        else: # Duplicate Subscription Jump
            amount = round(random.uniform(99.0, 299.0), 2)
            price_change_pct = round(random.uniform(1.20, 4.00), 4)
            merchant_rarity = round(random.uniform(0.2, 0.6), 2)
            
        records.append({
            "tx_id": f"TX_ANOM_{i:04d}",
            "raw_description": f"SUSPICIOUS {anom_type.upper()} {cat.upper()}",
            "category": cat,
            "amount": amount,
            "price_change_pct": price_change_pct,
            "merchant_rarity_score": merchant_rarity,
            "is_anomaly": 1,
            "anomaly_type": anom_type
        })
        
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Generating reproducible synthetic datasets (SEED=42)...")
    
    tx_df = generate_transaction_classification_dataset(samples_per_category=200)
    tx_file = os.path.join(DATA_DIR, "transactions_dataset.csv")
    tx_df.to_csv(tx_file, index=False)
    print(f"Saved {len(tx_df)} transaction records across {len(CATEGORIES)} categories to {tx_file}")
    
    expense_df = generate_monthly_expenses_dataset(num_users=10, months=36)
    expense_file = os.path.join(DATA_DIR, "monthly_expenses.csv")
    expense_df.to_csv(expense_file, index=False)
    print(f"Saved {len(expense_df)} monthly expense records to {expense_file}")
    
    anom_df = generate_anomalous_transactions_dataset(n_samples=500)
    anom_file = os.path.join(DATA_DIR, "anomalous_transactions.csv")
    anom_df.to_csv(anom_file, index=False)
    print(f"Saved {len(anom_df)} anomaly benchmark records to {anom_file}")
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
