from config import fetch_forecast_from_mssql
import pandas as pd

def load():
    df = fetch_forecast_from_mssql()
    df = df.dropna(subset=['ItemKey','SalesRepKey'])
    df = df[df['InvoiceAmount'] != 0]
    df = df[df['ItemKey'] == 5637145483]
    df['ItemKey'] = df['ItemKey'].astype('Int64').astype(str)
    df = df[df['CurrencyCode'].isin(['USD', 'EUR', 'INR','THB'])]
    eur_to_usd = 1.1
    inr_tousd = 0.012
    thb_to_usd = 0.031
    df.loc[df['CurrencyCode'] == 'EUR', 'InvoiceAmount'] *= eur_to_usd
    df.loc[df['CurrencyCode'] == 'INR', 'InvoiceAmount'] *= inr_tousd
    df.loc[df['CurrencyCode'] == 'THB', 'InvoiceAmount'] *= thb_to_usd
    df['CurrencyCode'] = 'USD'
    #df.to_csv('data.csv', index = False)
    #df['YearMonth'] = pd.to_datetime(df['InvoiceDate'], format = '%Y%M%D')
    df['Date'] = pd.to_datetime(df['InvoiceDate'])
    df = df.sort_values(['CustomerKey', 'ItemKey', 'Date'])
    # Drop rows where 'ItemKey' is missing
    # df = df.dropna(subset=['ItemKey'])

    # Group by (CustomerKey, ItemKey, YearMonth) to create time series

    monthly_sales = df.groupby(['Date']).agg({
        #'InvoiceAmount': 'sum',
        'Quantity': 'sum'
    }).reset_index()

    return monthly_sales

