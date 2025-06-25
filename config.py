import pyodbc
import pandas as pd
from sqlalchemy import create_engine,text

def insert_forecast_to_mssql(forecast_df):
    conn_str = create_engine(
        "mssql+pyodbc://U_user:abc123+@127.0.0.1:1433/AXDW"
        "?driver=ODBC+Driver+17+for+SQL+Server"
    )

    create_table_sql = """
        IF NOT EXISTS (
            SELECT * FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = 'FORECAST_SALEINVOICE'
        )
        BEGIN
            CREATE TABLE [dbo].[FORECAST_SALEINVOICE] (
                CustomerKey BIGINT,
                ItemKey BIGINT,
                YearMonth CHAR(6),
                FORECASTSALESAMOUNT FLOAT
            )
        END
        """

    # ใช้ engine.begin() เพื่อเปิด transaction
    with conn_str.begin() as conn:
        conn.execute(text(create_table_sql))

    forecast_df.to_sql(
        name='FORECAST_SALEINVOICE_2025',
        con=conn_str,
        if_exists='replace',                
        index=False                      
    )


def fetch_forecast_from_mssql():

    conn_str = (
        "mssql+pyodbc://U_user:abc123+@127.0.0.1:1433/AXDW"
        "?driver=ODBC+Driver+17+for+SQL+Server"
    )
    
    query = "SELECT * FROM [dbo].[FactSalesInvoiceLine4]"
    df = pd.read_sql(query, conn_str)
    
    return df



