import os
import pandas as pd
import numpy as np
import wrds


def open_connection():
    return wrds.Connection(wrds_username="varesl")


def get_initial_data(conn: wrds.Connection, cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["datadate"])

    # SQL query mimicking the PROC SQL step in SAS
    query = """
    SELECT 
        LEFT(REPLACE(cusip, ' ', ''), 6) AS cnum,
        c.gvkey,
        datadate,
        fyear,
        c.cik,
        LEFT(sic, 2) AS sic2,
        sic,
        naics,

        -- income statement
        sale, revt, cogs, xsga, dp, xrd, xad, ib, ebitda, ebit, nopi, spi, pi, txp, ni, txfed, txfo, txt, xint,

        -- cash flow and other
        capx, oancf, dvt, ob, gdwlia, gdwlip, gwo,

        -- assets
        rect, act, che, ppegt, invt, at, aco, intan, ao, ppent, gdwl, fatb, fatl,

        -- liabilities
        lct, dlc, dltt, lt, dm, dcvt, cshrc, dcpstk, pstk, ap, lco, lo, drc, drlt, txdi,

        -- equity and other
        ceq, scstkc, emp, csho,

        -- market
        ABS(prcc_f) AS prcc_f,
        csho * ABS(prcc_f) AS mve_f

    FROM comp.company c
    JOIN comp.funda f ON f.gvkey = c.gvkey
    WHERE
        at IS NOT NULL AND
        prcc_f IS NOT NULL AND
        ni IS NOT NULL AND
        datadate >= '1975-01-01' AND
        f.indfmt = 'INDL' AND
        f.datafmt = 'STD' AND
        f.popsrc = 'D' AND
        f.consol = 'C'
    """

    df = conn.raw_sql(query, date_cols=["datadate"])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


if __name__ == "__main__":
    cache_file = "cache/compustat_annual_init.csv"

    print("Opening connection to WRDS...")
    engine = open_connection()
    print("Connection opened.")

    print("Loading data (from cache if available)...")
    initial_data = get_initial_data(engine, cache_file)
    print("Data loaded.")

    print("Closing connection to WRDS...")
    engine.close()
