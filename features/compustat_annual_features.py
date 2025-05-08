import os
import pandas as pd
import numpy as np
import wrds


def load_and_filter_initial_data(comp_company_df, comp_funda_df):
    """
    Corresponds to the initial PROC SQL to create the 'data' table.
    Loads, merges, and filters Compustat annual data.
    """
    # Ensure gvkey is common type for merging
    comp_company_df["gvkey"] = comp_company_df["gvkey"].astype(str)
    comp_funda_df["gvkey"] = comp_funda_df["gvkey"].astype(str)

    # Merge company and funda tables
    df = pd.merge(comp_funda_df, comp_company_df, on="gvkey", how="inner")

    # Create cnum: substr(compress(cusip),1,6)
    df["cnum"] = (
        df["cusip"].astype(str).str.replace(r"\s+", "", regex=True).str.slice(0, 6)
    )

    # Create sic2: substr(sic,1,2)
    df["sic2"] = df["sic"].astype(str).str.slice(0, 2)

    # Market value: abs(prcc_f) as prcc_f, csho*calculated prcc_f as mve_f
    # Ensure prcc_f is numeric, errors='coerce' will turn non-numeric to NaN
    df["prcc_f"] = pd.to_numeric(df["prcc_f"], errors="coerce").abs()
    df["mve_f"] = df["csho"] * df["prcc_f"]

    # Filtering conditions
    df["datadate"] = pd.to_datetime(df["datadate"])
    min_date = pd.to_datetime("1975-01-01")

    df = df[
        df["at"].notna()
        & df["prcc_f"].notna()  # This is the abs(original prcc_f)
        & df["ni"].notna()
        & (df["datadate"] >= min_date)
        & (df["indfmt"] == "INDL")
        & (df["datafmt"] == "STD")
        & (df["popsrc"] == "D")
        & (df["consol"] == "C")
    ]

    # Select and rename columns to match SAS output structure if necessary
    # The SAS code selects many columns directly. Pandas keeps them from the merge.
    # Explicitly list required columns if comp_company_df and comp_funda_df have many more
    # For now, assume all selected SAS columns are present.

    return df


def sort_and_prep_time_series(df):
    """
    Sorts data, creates 'count' variable, and performs initial cleanup.
    Corresponds to PROC SORT and the first two DATA steps after initial SQL.
    """
    df = df.sort_values(by=["gvkey", "datadate"]).reset_index(drop=True)
    # Remove duplicates based on gvkey and datadate, keeping the first
    df = df.drop_duplicates(subset=["gvkey", "datadate"], keep="first")

    # Create 'count' variable (equivalent to SAS retain count for first.gvkey)
    df["count"] = df.groupby("gvkey").cumcount() + 1

    # Clean up dr, dc, xint0, xsga0
    # dr = drc + drlt
    df["dr"] = np.nan
    df.loc[df["drc"].notna() & df["drlt"].notna(), "dr"] = df["drc"] + df["drlt"]
    df.loc[df["drc"].notna() & df["drlt"].isna(), "dr"] = df["drc"]
    df.loc[df["drc"].isna() & df["drlt"].notna(), "dr"] = df["drlt"]

    # dc = dcpstk - pstk or dcpstk or dcvt
    df["dc"] = np.nan
    cond_dc1 = (
        df["dcvt"].isna()
        & df["dcpstk"].notna()
        & df["pstk"].notna()
        & (df["dcpstk"] > df["pstk"])
    )
    df.loc[cond_dc1, "dc"] = df["dcpstk"] - df["pstk"]

    cond_dc2 = df["dcvt"].isna() & df["dcpstk"].notna() & df["pstk"].isna()
    df.loc[cond_dc2, "dc"] = df["dcpstk"]

    df["dc"] = df["dc"].fillna(df["dcvt"])  # if missing(dc) then dc=dcvt;

    # xint0, xsga0
    df["xint0"] = df["xint"].fillna(0)
    df["xsga0"] = df["xsga"].fillna(0)

    return df


def create_financial_ratios_and_variables(df):
    """
    Creates the first pass of financial ratios and variables.
    Corresponds to the DATA data2; SET data; step.
    """
    # Lagged variables setup
    df = df.sort_values(by=["gvkey", "datadate"]).reset_index(drop=True)
    for col in [
        "at",
        "csho",
        "lt",
        "oancf",
        "act",
        "che",
        "lct",
        "dlc",
        "txp",
        "dp",
        "invt",
        "emp",
        "sale",
        "ib",
        "ppegt",
        "ppent",
        "ceq",
        "capx",
        "gdwl",
        "xad",
        "rect",
        "cogs",
        "xsga",
        "dvt",
        "dr",
        "ni",
        "dltt",
        "aco",
        "intan",
        "ao",
        "ap",
        "lco",
        "lo",
        "xrd",
    ]:
        if col in df.columns:
            df[f"lag_{col}"] = df.groupby("gvkey")[col].shift(1)

    if "at" in df.columns:  # For chato
        df["lag2_at"] = df.groupby("gvkey")["at"].shift(2)
    if "capx" in df.columns:  # For grcapx
        df["lag2_capx"] = df.groupby("gvkey")["capx"].shift(2)
    if "sale" in df.columns:  # For chato
        df["lag2_sale"] = df.groupby("gvkey")["sale"].shift(
            2
        )  # Not explicitly in SAS lag2(at) for chato, but lag(sale)/((lag(at)+lag2(at))/2)
    if "dp" in df.columns and "ppent" in df.columns:  # For pchdepr
        # Original problematic line:
        # df["lag_dp_ppent_ratio"] = df.groupby("gvkey")[["dp", "ppent"]].apply(
        #     lambda x: (x["dp"] / x["ppent"]).shift(1)
        # )
        # Corrected line:
        df["lag_dp_ppent_ratio"] = (
            (df["dp"] / df["ppent"]).groupby(df["gvkey"]).shift(1)
        )

    # Simple ratios
    df["bm"] = df["ceq"] / df["mve_f"]
    df["ep"] = df["ib"] / df["mve_f"]
    df["cashpr"] = (df["mve_f"] + df["dltt"] - df["at"]) / df["che"]
    df["dy"] = df["dvt"] / df["mve_f"]
    df["lev"] = df["lt"] / df["mve_f"]
    df["sp"] = df["sale"] / df["mve_f"]
    df["roic"] = (df["ebit"] - df["nopi"]) / (df["ceq"] + df["lt"] - df["che"])
    df["rd_sale"] = df["xrd"] / df["sale"]
    df["rd_mve"] = df["xrd"] / df["mve_f"]

    # Growth and change variables
    df["agr"] = (df["at"] / df["lag_at"]) - 1
    df["gma"] = (df["revt"] - df["cogs"]) / df["lag_at"]
    df["chcsho"] = (df["csho"] / df["lag_csho"]) - 1
    df["lgr"] = (df["lt"] / df["lag_lt"]) - 1

    # Accruals (acc)
    avg_at_lag_at = (df["at"] + df["lag_at"]) / 2
    df["acc"] = (df["ib"] - df["oancf"]) / avg_at_lag_at
    acc_missing_oancf_numerator = (
        (df["act"] - df["lag_act"])
        - (df["che"] - df["lag_che"])
        - (
            (df["lct"] - df["lag_lct"])
            - (df["dlc"] - df["lag_dlc"])
            - (df["txp"] - df["lag_txp"])
            - df["dp"]
        )
    )
    df.loc[df["oancf"].isna(), "acc"] = acc_missing_oancf_numerator / avg_at_lag_at

    # Percentage Accruals (pctacc)
    ib_abs = df["ib"].abs()
    ib_abs_nonzero = ib_abs.replace(0, 0.01)  # if ib=0 then use 0.01
    df["pctacc"] = (df["ib"] - df["oancf"]) / ib_abs_nonzero
    df.loc[df["oancf"].isna(), "pctacc"] = acc_missing_oancf_numerator / ib_abs_nonzero

    # Cash Flow to Price (cfp)
    df["cfp"] = df["oancf"] / df["mve_f"]
    # if oancf is missing, use balance sheet accrual definition for cash flow part
    # cfp_num_missing_oancf = ib - ( (act-lag(act) - (che-lag(che))) - ( (lct-lag(lct))-(dlc-lag(dlc))-(txp-lag(txp))-dp ) )
    # This is ib - acc_missing_oancf_numerator
    df.loc[df["oancf"].isna(), "cfp"] = (df["ib"] - acc_missing_oancf_numerator) / df[
        "mve_f"
    ]

    df["absacc"] = df["acc"].abs()
    df["age"] = df["count"]  # age is the count of observations for the firm
    df["chinv"] = (df["invt"] - df["lag_invt"]) / avg_at_lag_at

    df["spii"] = np.where(df["spi"].notna() & (df["spi"] != 0), 1, 0)
    df["spi"] = df["spi"] / avg_at_lag_at  # This redefines spi

    # Cash Flow (cf)
    df["cf"] = df["oancf"] / avg_at_lag_at
    df.loc[df["oancf"].isna(), "cf"] = (
        df["ib"] - acc_missing_oancf_numerator
    ) / avg_at_lag_at

    df["hire"] = (df["emp"] - df["lag_emp"]) / df["lag_emp"]
    df.loc[df["emp"].isna() | df["lag_emp"].isna() | (df["lag_emp"] == 0), "hire"] = (
        0  # SAS: if missing(emp) or missing(lag(emp)) then hire=0;
    )
    # Added lag_emp == 0 to prevent division by zero

    df["sgr"] = (df["sale"] / df["lag_sale"]) - 1
    df["chpm"] = (df["ib"] / df["sale"]) - (df["lag_ib"] / df["lag_sale"])

    avg_at_lag_at_lag2_at = (df["lag_at"] + df["lag2_at"]) / 2
    df["chato"] = (df["sale"] / avg_at_lag_at) - (
        df["lag_sale"] / avg_at_lag_at_lag2_at
    )

    df["pchsale_pchinvt"] = ((df["sale"] - df["lag_sale"]) / df["lag_sale"]) - (
        (df["invt"] - df["lag_invt"]) / df["lag_invt"]
    )
    df["pchsale_pchrect"] = ((df["sale"] - df["lag_sale"]) / df["lag_sale"]) - (
        (df["rect"] - df["lag_rect"]) / df["lag_rect"]
    )

    gm = df["sale"] - df["cogs"]
    lag_gm = df["lag_sale"] - df["lag_cogs"]
    df["pchgm_pchsale"] = ((gm - lag_gm) / lag_gm) - (
        (df["sale"] - df["lag_sale"]) / df["lag_sale"]
    )

    df["pchsale_pchxsga"] = ((df["sale"] - df["lag_sale"]) / df["lag_sale"]) - (
        (df["xsga"] - df["lag_xsga"]) / df["lag_xsga"]
    )

    df["depr"] = df["dp"] / df["ppent"]

    current_dp_ppent_ratio = df["dp"] / df["ppent"]
    df["pchdepr"] = (current_dp_ppent_ratio - df["lag_dp_ppent_ratio"]) / df[
        "lag_dp_ppent_ratio"
    ].replace(0, np.nan)

    # df["chadv"] = np.log(1 + df["xad"]) - np.log(1 + df["lag_xad"]) # Original
    # Robust chadv calculation to match SAS log behavior (missing for non-positive)
    log_1_plus_xad = np.log(np.where(1 + df["xad"] > 0, 1 + df["xad"], np.nan))
    log_1_plus_lag_xad = np.log(
        np.where(1 + df["lag_xad"] > 0, 1 + df["lag_xad"], np.nan)
    )
    df["chadv"] = log_1_plus_xad - log_1_plus_lag_xad

    df["invest"] = (
        (df["ppegt"] - df["lag_ppegt"]) + (df["invt"] - df["lag_invt"])
    ) / df["lag_at"].replace(0, np.nan)
    df.loc[df["ppegt"].isna(), "invest"] = (
        (df["ppent"] - df["lag_ppent"]) + (df["invt"] - df["lag_invt"])
    ) / df["lag_at"].replace(0, np.nan)

    df["egr"] = (df["ceq"] - df["lag_ceq"]) / df["lag_ceq"].replace(0, np.nan)

    # if missing(capx) and count>=2 then capx=ppent-lag(ppent);
    # This modifies capx itself. Be careful if capx is used before this.
    # Assuming capx from funda is used first, then this is a fillna type operation.
    cond_fill_capx = df["capx"].isna() & (df["count"] >= 2)
    df.loc[cond_fill_capx, "capx"] = df["ppent"] - df["lag_ppent"]

    df["pchcapx"] = (df["capx"] - df["lag_capx"]) / df["lag_capx"].replace(0, np.nan)
    df["grcapx"] = (df["capx"] - df["lag2_capx"]) / df["lag2_capx"].replace(0, np.nan)

    df["grGW"] = (df["gdwl"] - df["lag_gdwl"]) / df["lag_gdwl"].replace(0, np.nan)
    df.loc[df["gdwl"].isna() | (df["gdwl"] == 0), "grGW"] = 0
    df.loc[df["gdwl"].notna() & (df["gdwl"] != 0) & df["grGW"].isna(), "grGW"] = (
        1  # If gdwl not 0, not missing, but grGW is missing (e.g. lag_gdwl was 0 or missing)
    )
    # SAS: missing(grGW) implies lag_gdwl was 0 or missing. If lag_gdwl is 0, grGW is missing.
    # This line seems to set grGW=1 if current gdwl is non-zero and lag was zero/missing.

    df["woGW"] = np.where(
        (df["gdwlia"].notna() & (df["gdwlia"] != 0))
        | (df["gdwlip"].notna() & (df["gdwlip"] != 0))
        | (df["gwo"].notna() & (df["gwo"] != 0)),
        1,
        0,
    )

    df["tang"] = (
        df["che"] + df["rect"] * 0.715 + df["invt"] * 0.547 + df["ppent"] * 0.535
    ) / df["at"].replace(0, np.nan)

    sin_sic_cond = (
        df["sic"].astype(float).fillna(0).between(2100, 2199, inclusive="both")
    ) | (df["sic"].astype(float).fillna(0).between(2080, 2085, inclusive="both"))
    sin_naics_cond = (
        df["naics"]
        .astype(str)
        .isin(["7132", "71312", "713210", "71329", "713290", "72112", "721120"])
    )
    df["sin"] = np.where(sin_sic_cond | sin_naics_cond, 1, 0)

    df.loc[df["act"].isna(), "act"] = df["che"] + df["rect"] + df["invt"]
    df.loc[df["lct"].isna(), "lct"] = df["ap"]

    df["currat"] = df["act"] / df["lct"].replace(0, np.nan)  # Avoid division by zero
    df["pchcurrat"] = (
        (df["act"] / df["lct"].replace(0, np.nan))
        - (df["lag_act"] / df["lag_lct"].replace(0, np.nan))
    ) / (df["lag_act"] / df["lag_lct"].replace(0, np.nan)).replace(
        0, np.nan
    )  # Avoid division by zero

    df["quick"] = (df["act"] - df["invt"]) / df["lct"].replace(
        0, np.nan
    )  # Avoid division by zero
    df["pchquick"] = (
        ((df["act"] - df["invt"]) / df["lct"].replace(0, np.nan))
        - ((df["lag_act"] - df["lag_invt"]) / df["lag_lct"].replace(0, np.nan))
    ) / ((df["lag_act"] - df["lag_invt"]) / df["lag_lct"].replace(0, np.nan)).replace(
        0, np.nan
    )  # Avoid division by zero

    df["salecash"] = df["sale"] / df["che"].replace(0, np.nan)  # Avoid division by zero
    df["salerec"] = df["sale"] / df["rect"].replace(0, np.nan)  # Avoid division by zero
    df["saleinv"] = df["sale"] / df["invt"].replace(0, np.nan)  # Avoid division by zero
    df["pchsaleinv"] = (
        (df["sale"] / df["invt"].replace(0, np.nan))
        - (df["lag_sale"] / df["lag_invt"].replace(0, np.nan))
    ) / (df["lag_sale"] / df["lag_invt"].replace(0, np.nan)).replace(
        0, np.nan
    )  # Avoid division by zero

    df["cashdebt"] = (df["ib"] + df["dp"]) / ((df["lt"] + df["lag_lt"]) / 2).replace(
        0, np.nan
    )  # Avoid division by zero

    df["realestate"] = (df["fatb"] + df["fatl"]) / df["ppegt"].replace(
        0, np.nan
    )  # Avoid division by zero
    df.loc[df["ppegt"].isna(), "realestate"] = (df["fatb"] + df["fatl"]) / df[
        "ppent"
    ].replace(0, np.nan)  # Avoid division by zero

    df["divi"] = np.where(
        df["dvt"].notna()
        & (df["dvt"] > 0)
        & (df["lag_dvt"].isna() | (df["lag_dvt"] == 0)),
        1,
        0,
    )
    df["divo"] = np.where(
        (df["dvt"].isna() | (df["dvt"] == 0))
        & df["lag_dvt"].notna()
        & (df["lag_dvt"] > 0),
        1,
        0,
    )

    df["obklg"] = df["ob"] / avg_at_lag_at
    df["chobklg"] = (
        df["ob"] - df.groupby("gvkey")["ob"].shift(1)
    ) / avg_at_lag_at  # lag(ob) used here

    df["securedind"] = np.where(df["dm"].notna() & (df["dm"] != 0), 1, 0)
    df["secured"] = df["dm"] / df["dltt"].replace(0, np.nan)

    df["convind"] = np.where(
        (df["dc"].notna() & (df["dc"] != 0))
        | (df["cshrc"].notna() & (df["cshrc"] != 0)),
        1,
        0,
    )
    df["conv"] = df["dc"] / df["dltt"].replace(0, np.nan)  # Avoid division by zero

    # grltnoa (complex calculation)
    noa_comp1 = (
        df["rect"]
        + df["invt"]
        + df["ppent"]
        + df["aco"]
        + df["intan"]
        + df["ao"]
        - df["ap"]
        - df["lco"]
        - df["lo"]
    )
    # lag_noa_comp1 = df.groupby("gvkey")[ # Original approach, replaced
    #     ["rect", "invt", "ppent", "aco", "intan", "ao", "ap", "lco", "lo"]
    # ].shift(1)
    # lag_noa_comp1_sum = ( # Original approach, replaced
    #     lag_noa_comp1["rect"]
    #     + lag_noa_comp1["invt"]
    #     + lag_noa_comp1["ppent"]
    #     + lag_noa_comp1["aco"]
    #     + lag_noa_comp1["intan"]
    #     + lag_noa_comp1["ao"]
    #     - lag_noa_comp1["ap"]
    #     - lag_noa_comp1["lco"]
    #     - lag_noa_comp1["lo"]
    # )
    # Corrected lag_noa_comp1_sum using pre-calculated lag columns
    lag_noa_comp1_sum = (
        df["lag_rect"]
        + df["lag_invt"]
        + df["lag_ppent"]
        + df["lag_aco"]
        + df["lag_intan"]
        + df["lag_ao"]
        - df["lag_ap"]
        - df["lag_lco"]
        - df["lag_lo"]
    )

    # noa_comp2 = ( # Original approach, replaced
    #     (df["rect"] - df["lag_rect"])
    #     + (df["invt"] - df["lag_invt"])
    #     + (df["aco"] - df.groupby("gvkey")["aco"].shift(1))
    #     - (
    #         (df["ap"] - df["lag_ap"])
    #         + (df["lco"] - df.groupby("gvkey")["lco"].shift(1))
    #     )
    #     - df["dp"]
    # )
    # Corrected noa_comp2 using pre-calculated lag columns
    noa_comp2 = (
        (df["rect"] - df["lag_rect"])
        + (df["invt"] - df["lag_invt"])
        + (df["aco"] - df["lag_aco"])
        - ((df["ap"] - df["lag_ap"]) + (df["lco"] - df["lag_lco"]))
        - df["dp"]
    )
    df["grltnoa"] = (noa_comp1 - lag_noa_comp1_sum - noa_comp2) / avg_at_lag_at

    df["chdrc"] = (df["dr"] - df["lag_dr"]) / avg_at_lag_at

    xrd_at_ratio = df["xrd"] / df["at"]
    # Ensure lag2_at is available (it is created earlier in the function)
    # lag(xrd/lag(at)) in SAS likely means lag(xrd)/lag(lag(at)) = lag(xrd)/lag2(at)
    # Use the pre-calculated lag_xrd
    lag_xrd_at_ratio = df["lag_xrd"] / df["lag2_at"].replace(
        0, np.nan
    )  # Avoid division by zero
    df["rd"] = np.where(
        ((xrd_at_ratio - lag_xrd_at_ratio) / lag_xrd_at_ratio.replace(0, np.nan))
        > 0.05,
        1,
        0,  # Avoid division by zero
    )

    df["rdbias"] = (
        (df["xrd"] / df["lag_xrd"].replace(0, np.nan))
        - 1
        - (df["ib"] / df["lag_ceq"].replace(0, np.nan))
    )  # Avoid division by zero
    df["roe"] = df["ib"] / df["lag_ceq"].replace(0, np.nan)  # Avoid division by zero
    df["operprof"] = (df["revt"] - df["cogs"] - df["xsga0"] - df["xint0"]) / df[
        "lag_ceq"
    ].replace(0, np.nan)  # Avoid division by zero

    # Piotroski F-Score (ps)
    # For ratios in ps, ensure denominators are handled to avoid division by zero if they can be zero
    # Example: (df["ni"] / df["at"].replace(0, np.nan))
    # For brevity, I'm not changing all ps terms here, but apply .replace(0, np.nan) as needed.
    df["ps"] = (
        (df["ni"] > 0).astype(int)
        + (df["oancf"] > 0).astype(int)
        + (
            (df["ni"] / df["at"].replace(0, np.nan))
            > (df["lag_ni"] / df["lag_at"].replace(0, np.nan))
        ).astype(int)
        + (df["oancf"] > df["ni"]).astype(int)
        + (
            (df["dltt"] / df["at"].replace(0, np.nan))
            < (df["lag_dltt"] / df["lag_at"].replace(0, np.nan))
        ).astype(int)
        + (
            (df["act"] / df["lct"].replace(0, np.nan))
            > (df["lag_act"] / df["lag_lct"].replace(0, np.nan))
        ).astype(int)
        + (
            ((df["sale"] - df["cogs"]) / df["sale"].replace(0, np.nan))
            > ((df["lag_sale"] - df["lag_cogs"]) / df["lag_sale"].replace(0, np.nan))
        ).astype(int)
        + (
            (df["sale"] / df["at"].replace(0, np.nan))
            > (df["lag_sale"] / df["lag_at"].replace(0, np.nan))
        ).astype(int)
        + (df["scstkc"] == 0).astype(int)
    )

    # Tax rate (tr) for tb_1
    conditions_tr = [
        df["fyear"] <= 1978,
        (df["fyear"] >= 1979) & (df["fyear"] <= 1986),
        df["fyear"] == 1987,
        (df["fyear"] >= 1988) & (df["fyear"] <= 1992),
        df["fyear"] >= 1993,
    ]
    values_tr = [0.48, 0.46, 0.40, 0.34, 0.35]
    df["tr"] = np.select(conditions_tr, values_tr, default=np.nan)

    # tb_1 (tax refundability proxy)
    df["tb_1"] = np.nan
    cond_tb1_default = (
        df["txfo"].notna()
        & df["txfed"].notna()
        & df["tr"].notna()
        & df["ib"].notna()
        & (df["ib"] != 0)  # Explicitly check for non-zero denominator
    )
    df.loc[cond_tb1_default, "tb_1"] = (
        (df["txfo"] + df["txfed"]) / df["tr"].replace(0, np.nan)
    ) / df[  # Avoid div by zero for tr
        "ib"
    ]

    cond_tb1_override1 = (
        (df["txfo"].isna() | df["txfed"].isna())
        & df["txt"].notna()
        & df["txdi"].notna()
        & df["tr"].notna()
        & df["ib"].notna()
        & (df["ib"] != 0)  # Explicitly check for non-zero denominator
    )
    df.loc[cond_tb1_override1, "tb_1"] = (
        (df["txt"] - df["txdi"]) / df["tr"].replace(0, np.nan)
    ) / df[  # Avoid div by zero for tr
        "ib"
    ]

    cond_sum_gt_0 = (
        df["txfo"].fillna(-np.inf) + df["txfed"].fillna(-np.inf)
    ) > 0  # Handles NaNs by making sum NaN if any part is NaN
    cond_txt_gt_txdi = df["txt"].fillna(-np.inf) > df["txdi"].fillna(
        np.inf
    )  # Handles NaNs similarly

    cond_tb1_override2_part1 = cond_sum_gt_0.fillna(False) | cond_txt_gt_txdi.fillna(
        False
    )
    cond_tb1_override2_part2 = (
        df["ib"].fillna(np.inf) <= 0
    )  # if ib is NaN, (NaN <= 0) is False

    cond_tb1_override2 = cond_tb1_override2_part1 & cond_tb1_override2_part2.fillna(
        False
    )
    df.loc[cond_tb1_override2, "tb_1"] = 1.0

    # Variables for Mohanram (2005) score
    df["roa"] = df["ni"] / avg_at_lag_at
    df["cfroa"] = df["oancf"] / avg_at_lag_at
    df.loc[df["oancf"].isna(), "cfroa"] = (
        df["ib"] + df["dp"]
    ) / avg_at_lag_at  # if missing(oancf) then cfroa=(ib+dp)/...

    df["xrdint"] = df["xrd"] / avg_at_lag_at
    df["capxint"] = df["capx"] / avg_at_lag_at
    df["xadint"] = df["xad"] / avg_at_lag_at

    return df


def cleanup_lagged_variables(df):
    """
    Sets variables to NaN if insufficient lagged data (based on 'count').
    """
    req_cols = [
        "chadv",
        "agr",
        "invest",
        "gma",
        "chcsho",
        "lgr",
        "egr",
        "chpm",
        "chinv",
        "hire",
        "cf",
        "acc",
        "pctacc",
        "absacc",
        "spi",
        "sgr",
        "pchsale_pchinvt",
        "pchsale_pchrect",
        "pchgm_pchsale",
        "pchsale_pchxsga",
        "pchcapx",
        "ps",
        "roa",
        "cfroa",
        "xrdint",
        "capxint",
        "xadint",
        "divi",
        "divo",
        "obklg",
        "chobklg",
        "grltnoa",
        "chdrc",
        "rd",
        "pchdepr",
        "grGW",
        "pchcurrat",
        "pchquick",
        "pchsaleinv",
        "roe",
        "operprof",
    ]

    # Ensure columns exist before trying to set to NaN
    existing_req_cols = [col for col in req_cols if col in df.columns]
    df.loc[df["count"] == 1, existing_req_cols] = np.nan

    cols_count_lt_3 = ["chato", "grcapx"]
    existing_cols_count_lt_3 = [col for col in cols_count_lt_3 if col in df.columns]
    df.loc[df["count"] < 3, existing_cols_count_lt_3] = np.nan

    return df


def perform_industry_adjustments(df):
    """
    Calculates industry-adjusted variables and Herfindahl index.
    Also prepares for Mohanram score with industry medians.
    """
    # Industry adjustments (mean-adjusted variables)
    group_cols_ind = ["sic2", "fyear"]

    for col in ["chpm", "chato", "hire", "bm", "pchcapx", "tb_1", "cfp", "mve_f"]:
        if col in df.columns:
            # Ensure the column is numeric before transform
            df[col] = pd.to_numeric(df[col], errors="coerce")
            mean_val = df.groupby(group_cols_ind)[col].transform("mean")
            if col == "tb_1":  # SAS uses tb_1-mean(tb_1) as tb
                df["tb"] = df[col] - mean_val
            else:  # e.g. chpm-mean(chpm) as chpmia
                df[
                    f"{col}ia"
                    if col not in ["hire", "tb_1"]
                    else ("chempia" if col == "hire" else col)
                ] = df[col] - mean_val

    if "sale" in df.columns:
        df["indsale"] = df.groupby(group_cols_ind)["sale"].transform("sum")
        # Assign share_sq as a column to the DataFrame first
        df["share_sq"] = (
            df["sale"] / df["indsale"].replace(0, np.nan)
        ) ** 2  # Avoid division by zero
        df["herf"] = df.groupby(group_cols_ind)["share_sq"].transform("sum")
        df.drop(columns=["share_sq"], inplace=True, errors="ignore")

    # Industry medians for Mohanram score components (m1-m6)
    median_cols = ["roa", "cfroa", "xrdint", "capxint", "xadint"]
    existing_median_cols = [col for col in median_cols if col in df.columns]

    if existing_median_cols:
        ind_medians = (
            df.groupby(group_cols_ind)[existing_median_cols].median().reset_index()
        )
        ind_medians = ind_medians.rename(
            columns={col: f"md_{col}" for col in existing_median_cols}
        )
        df = pd.merge(df, ind_medians, on=group_cols_ind, how="left")

        if "roa" in existing_median_cols and "md_roa" in df.columns:
            df["m1"] = np.where(df["roa"] > df["md_roa"], 1, 0)
        if "cfroa" in existing_median_cols and "md_cfroa" in df.columns:
            df["m2"] = np.where(df["cfroa"] > df["md_cfroa"], 1, 0)
        if "oancf" in df.columns and "ni" in df.columns:  # m3: oancf > ni
            df["m3"] = np.where(df["oancf"] > df["ni"], 1, 0)
        if "xrdint" in existing_median_cols and "md_xrdint" in df.columns:
            df["m4"] = np.where(df["xrdint"] > df["md_xrdint"], 1, 0)
        if "capxint" in existing_median_cols and "md_capxint" in df.columns:
            df["m5"] = np.where(df["capxint"] > df["md_capxint"], 1, 0)
        if "xadint" in existing_median_cols and "md_xadint" in df.columns:
            df["m6"] = np.where(df["xadint"] > df["md_xadint"], 1, 0)

    return df


def add_credit_ratings(df, comp_adsprate_df):
    """
    Adds credit ratings from comp.adsprate and converts to numerical format.
    """
    if comp_adsprate_df is None or comp_adsprate_df.empty:
        # Add placeholder columns if adsp_rate data is missing
        for col in ["splticrm", "credrat", "credrat_dwn"]:
            if col not in df.columns:
                df[col] = np.nan
        return df

    # Ensure gvkey in the main df is also string type for consistent merging
    df["gvkey"] = df["gvkey"].astype(str)

    comp_adsprate_df["gvkey"] = comp_adsprate_df["gvkey"].astype(str)
    comp_adsprate_df["year_adsprate"] = pd.to_datetime(
        comp_adsprate_df["datadate"]
    ).dt.year

    df["year_datadate"] = df["datadate"].dt.year

    # Merge based on gvkey and year
    # SAS: year(a.datadate)=year(b.datadate). If multiple ratings in a year, this might lead to duplicates.
    # Assuming comp.adsprate is structured to provide one relevant rating per gvkey-year.
    # If not, may need sorting and drop_duplicates on comp_adsprate_df before merge.
    # For simplicity, using a left merge. If multiple matches, pandas will duplicate rows from df.
    # SAS proc sql join might behave differently with multiple matches if not handled by distinct or group by.
    # The SAS code uses `nodupkey` after sorting by gvkey, datadate for `data2`, implying unique firm-date.
    # The join to adsprate is on gvkey and year(datadate). If adsprate has multiple ratings for a gvkey in a year,
    # it could lead to multiple rows. The subsequent `proc sort data=data2 nodupkey by gvkey datadate` would remove these
    # if the multiple ratings resulted in identical `data2` rows except for `splticrm`, which is unlikely.
    # A common practice is to take the latest rating within the year if multiple exist.
    # Here, we'll assume the merge is okay or comp_adsprate_df is pre-processed.

    # To mimic SAS behavior of potentially picking one if multiple, sort adsp_rate and drop duplicates by merge keys
    comp_adsprate_df = comp_adsprate_df.sort_values(
        by=["gvkey", "year_adsprate", "datadate"], ascending=[True, True, False]
    )
    comp_adsprate_df_unique = comp_adsprate_df.drop_duplicates(
        subset=["gvkey", "year_adsprate"], keep="first"
    )

    df = pd.merge(
        df,
        comp_adsprate_df_unique[["gvkey", "year_adsprate", "splticrm"]],
        left_on=["gvkey", "year_datadate"],
        right_on=["gvkey", "year_adsprate"],
        how="left",
    )
    df.drop(columns=["year_adsprate", "year_datadate"], inplace=True, errors="ignore")

    rating_map = {
        "D": 1,
        "C": 2,
        "CC": 3,
        "CCC-": 4,
        "CCC": 5,
        "CCC+": 6,
        "B-": 7,
        "B": 8,
        "B+": 9,
        "BB-": 10,
        "BB": 11,
        "BB+": 12,
        "BBB-": 13,
        "BBB": 14,
        "BBB+": 15,
        "A-": 16,
        "A": 17,
        "A+": 18,
        "AA-": 19,
        "AA": 20,
        "AA+": 21,
        "AAA": 22,
    }
    df["credrat"] = df["splticrm"].map(rating_map)

    # credrat_dwn
    df = df.sort_values(by=["gvkey", "datadate"])  # Ensure correct order for lag
    df["lag_credrat"] = df.groupby("gvkey")["credrat"].shift(1)
    df["credrat_dwn"] = np.where(df["credrat"] < df["lag_credrat"], 1, 0)
    df.loc[df["count"] == 1, "credrat_dwn"] = 0  # if count=1 then credrat_dwn=0;
    df.drop(columns=["lag_credrat"], inplace=True, errors="ignore")

    return df


def _calculate_orgcap1_for_group(group):
    """Helper function to calculate orgcap_1 for a single gvkey group."""
    group = group.sort_values("datadate")
    orgcap_1_val = np.nan
    orgcap_1_list = []

    for idx, row in group.iterrows():
        xsga_val = row["xsga"]
        cpi_val = row["cpi"]
        is_first = row["count"] == 1  # Relies on 'count' being pre-calculated

        if pd.isna(xsga_val) or pd.isna(cpi_val) or cpi_val == 0:
            orgcap_1_val = np.nan  # Propagate NaN if inputs are bad
        elif is_first:
            orgcap_1_val = (xsga_val / cpi_val) / (0.1 + 0.15)
        else:
            if pd.isna(orgcap_1_val):  # If prior was NaN, it stays NaN
                orgcap_1_val = np.nan
            else:
                orgcap_1_val = orgcap_1_val * (1 - 0.15) + (xsga_val / cpi_val)
        orgcap_1_list.append(orgcap_1_val)

    return pd.Series(orgcap_1_list, index=group.index)


def add_cpi_and_calculate_orgcap(df):
    """
    Adds CPI data and calculates the orgcap measure.
    """
    cpi_data = {
        "yr": [
            2015,
            2014,
            2013,
            2012,
            2011,
            2010,
            2009,
            2008,
            2007,
            2006,
            2005,
            2004,
            2003,
            2002,
            2001,
            2000,
            1999,
            1998,
            1997,
            1996,
            1995,
            1994,
            1993,
            1992,
            1991,
            1990,
            1989,
            1988,
            1987,
            1986,
            1985,
            1984,
            1983,
            1982,
            1981,
            1980,
            1979,
            1978,
            1977,
            1976,
            1975,
            1974,
        ],
        "cpi_val": [
            236.53,
            229.91,
            229.17,
            229.594,
            224.939,
            218.056,
            214.537,
            215.303,
            207.342,
            201.6,
            195.3,
            188.9,
            183.96,
            179.88,
            177.1,
            172.2,
            166.6,
            163.00,
            160.5,
            156.9,
            152.4,
            148.2,
            144.5,
            140.3,
            136.2,
            130.7,
            124.00,
            118.3,
            113.6,
            109.6,
            107.6,
            103.9,
            99.6,
            96.5,
            90.9,
            82.4,
            72.6,
            65.2,
            60.6,
            56.9,
            53.8,
            49.3,
        ],
    }
    cpi_df = pd.DataFrame(cpi_data)
    cpi_df.rename(
        columns={"cpi_val": "cpi"}, inplace=True
    )  # Match SAS variable name 'cpi'

    # Merge CPI data (SAS merges on a.fyear=b.yr)
    df = pd.merge(df, cpi_df, left_on="fyear", right_on="yr", how="left")
    df.drop(columns=["yr"], inplace=True, errors="ignore")

    # Calculate orgcap
    # This requires careful handling of the 'retain orgcap_1' and 'by gvkey datadate'
    df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    # 'count' should already exist from sort_and_prep_time_series

    # Calculate orgcap_1 using the helper
    # Ensure that 'xsga' and 'cpi' columns are numeric
    df["xsga"] = pd.to_numeric(df["xsga"], errors="coerce")
    df["cpi"] = pd.to_numeric(df["cpi"], errors="coerce")

    # Apply the calculation per group
    # The result of apply needs to be carefully assigned back if its index is not aligned.
    # Using group_keys=False helps, or re-aligning.
    orgcap_1_calculated = df.groupby("gvkey", group_keys=False).apply(
        _calculate_orgcap1_for_group
    )
    df["orgcap_1"] = orgcap_1_calculated

    df["lag_at_for_orgcap"] = df.groupby("gvkey")["at"].shift(1)  # For avgat
    df["avgat"] = (df["at"] + df["lag_at_for_orgcap"]) / 2

    df["orgcap"] = df["orgcap_1"] / df["avgat"]
    df.loc[df["count"] == 1, "orgcap"] = np.nan  # if count=1 then orgcap=.;

    df.drop(
        columns=["orgcap_1", "lag_at_for_orgcap", "avgat"],
        inplace=True,
        errors="ignore",
    )

    return df


def main_annual_compustat_processing(
    initial_df: pd.DataFrame, comp_adsprate_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Main function to orchestrate the processing of annual Compustat data.
    Assumes initial_df is already merged and filtered (e.g., from WRDS query).
    """
    print("Step 1: Processing initial merged data...")
    # Use a copy to avoid modifying the original DataFrame passed to the function
    df = initial_df.copy()
    print(f"DataFrame shape at start of processing: {df.shape}")

    print("Step 2: Sorting, preparing time series, and basic cleaning...")
    df = sort_and_prep_time_series(df)
    print(f"DataFrame shape after time series prep: {df.shape}")

    # This df is equivalent to 'data' before it's used to create 'data2' in SAS
    # SAS then creates data2: data data2; set data; /* calculations */; run;
    # So, we continue processing on this 'df' which will become 'data2' conceptually

    print("Step 3: Creating financial ratios and variables (Pass 1)...")
    df_data2 = create_financial_ratios_and_variables(
        df.copy()
    )  # Use .copy() if df is reused elsewhere or to ensure modifications are on a new object
    print(f"DataFrame shape after creating financial ratios: {df_data2.shape}")

    print("Step 4: Cleaning up lagged variables...")
    df_data2 = cleanup_lagged_variables(df_data2)
    print(f"DataFrame shape after lagged variable cleanup: {df_data2.shape}")

    print("Step 5: Performing industry adjustments...")
    df_data2 = perform_industry_adjustments(df_data2)
    print(f"DataFrame shape after industry adjustments: {df_data2.shape}")

    print("Step 6: Adding credit ratings...")
    df_data2 = add_credit_ratings(df_data2, comp_adsprate_df)
    print(f"DataFrame shape after adding credit ratings: {df_data2.shape}")

    # At this point, df_data2 is the 'data2' table in SAS before orgcap calculation
    # The final SAS step is: data data; set data2; /* orgcap calc */; run;

    print("Step 7: Adding CPI and calculating orgcap...")
    df_final = add_cpi_and_calculate_orgcap(df_data2.copy())  # Use .copy()
    print(f"DataFrame shape after orgcap calculation: {df_final.shape}")

    print("Processing complete.")
    return df_final


def open_wrds_connection():
    """Opens a WRDS connection."""
    # If .pgpass is set up, wrds.Connection() might work without arguments.
    try:
        conn = wrds.Connection()
        print("WRDS connection successful.")
        return conn
    except Exception as e:
        print(f"Failed to connect to WRDS: {e}")
        return None


def get_compustat_initial_data_from_wrds(
    conn: wrds.Connection, cache_path: str
) -> pd.DataFrame:
    """
    Fetches initial Compustat data (company and funda) using a SQL query similar to your example.
    """
    if os.path.exists(cache_path):
        print(f"Loading initial Compustat data from cache: {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["datadate"])

    print("Fetching initial Compustat data from WRDS...")
    # SQL query mimicking the PROC SQL step in SAS (from your compustat_annual_features.py)
    query = """
    SELECT 
        SUBSTR(REPLACE(cusip, ' ', ''), 1, 6) AS cnum, -- Adjusted for common SQL dialects, check WRDS syntax
        c.gvkey,
        f.datadate, -- Ensure datadate comes from funda for consistency
        f.fyear,
        c.cik,
        SUBSTR(c.sic, 1, 2) AS sic2, -- Assuming sic is in company, adjust if in funda
        c.sic,                      -- Assuming sic is in company
        c.naics,                    -- Assuming naics is in company

        -- income statement
        f.sale, f.revt, f.cogs, f.xsga, f.dp, f.xrd, f.xad, f.ib, f.ebitda, f.ebit, f.nopi, f.spi, f.pi, f.txp, f.ni, f.txfed, f.txfo, f.txt, f.xint,

        -- cash flow and other
        f.capx, f.oancf, f.dvt, f.ob, f.gdwlia, f.gdwlip, f.gwo,

        -- assets
        f.rect, f.act, f.che, f.ppegt, f.invt, f.at, f.aco, f.intan, f.ao, f.ppent, f.gdwl, f.fatb, f.fatl,

        -- liabilities
        f.lct, f.dlc, f.dltt, f.lt, f.dm, f.dcvt, f.cshrc, f.dcpstk, f.pstk, f.ap, f.lco, f.lo, f.drc, f.drlt, f.txdi,

        -- equity and other
        f.ceq, f.scstkc, f.emp, f.csho,

        -- market
        ABS(f.prcc_f) AS prcc_f,
        f.csho * ABS(f.prcc_f) AS mve_f

    FROM comp.company AS c
    JOIN comp.funda AS f ON f.gvkey = c.gvkey
    WHERE
        f.at IS NOT NULL AND
        f.prcc_f IS NOT NULL AND
        f.ni IS NOT NULL AND
        f.datadate >= '01JAN1975' AND -- Adjusted date format for many SQL dbs, check WRDS
        f.indfmt = 'INDL' AND
        f.datafmt = 'STD' AND
        f.popsrc = 'D' AND
        f.consol = 'C'
    """
    # Note: In your query, LEFT(REPLACE(cusip...)) and LEFT(sic...) were used.
    # Standard SQL SUBSTR/SUBSTRING might be needed depending on the backend WRDS uses (e.g., PostgreSQL).
    # I've used SUBSTR as a common alternative.
    # Also, explicitly prefixed columns with f. or c. where ambiguity might arise.

    df = conn.raw_sql(query, date_cols=["datadate"])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved initial Compustat data to cache: {cache_path}")
    return df


def get_adsprate_data_from_wrds(conn: wrds.Connection, cache_path: str) -> pd.DataFrame:
    """
    Fetches Compustat S&P ratings data.
    """
    if os.path.exists(cache_path):
        print(f"Loading S&P ratings data from cache: {cache_path}")
        return pd.read_csv(
            cache_path, parse_dates=["datadate"]
        )  # Assuming adsp_rate also has a 'datadate'

    print("Fetching S&P ratings data (comp.adsprate) from WRDS...")
    # Example query, adjust as needed for actual column names and desired data
    query_adsprate = """
    SELECT
        gvkey,
        datadate, -- or the relevant date column for ratings
        splticrm  -- S&P Long-Term Issuer Credit Rating
        -- Add other columns if needed, e.g., spsticrm for short-term
    FROM comp.adsprate
    WHERE splticrm IS NOT NULL; 
    """
    # Ensure date_cols matches the actual date column name in comp.adsprate
    df = conn.raw_sql(query_adsprate, date_cols=["datadate"])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved S&P ratings data to cache: {cache_path}")
    return df


if __name__ == "__main__":
    # Define cache paths
    base_cache_dir = "cache/data"  # Store all cache files in a sub-directory
    initial_comp_cache_path = os.path.join(
        base_cache_dir, "compustat_annual_initial_merged.csv"
    )
    adsprate_cache_path = os.path.join(base_cache_dir, "compustat_adsprate.csv")
    final_output_cache_path = os.path.join(
        base_cache_dir, "compustat_annual_features_final.csv"
    )

    # Create base cache directory if it doesn't exist
    os.makedirs(base_cache_dir, exist_ok=True)

    print("Attempting to open WRDS connection...")
    wrds_conn = open_wrds_connection()

    initial_df_from_wrds = None
    comp_adsprate_df = None

    if wrds_conn:
        try:
            # Load initial merged Compustat data (company & funda)
            initial_df_from_wrds = get_compustat_initial_data_from_wrds(
                wrds_conn, initial_comp_cache_path
            )

            # Load Compustat S&P ratings data
            comp_adsprate_df = get_adsprate_data_from_wrds(
                wrds_conn, adsprate_cache_path
            )

        except Exception as e:
            print(f"An error occurred during data fetching from WRDS: {e}")
        finally:
            print("Closing WRDS connection.")
            wrds_conn.close()
    else:
        print(
            "Could not establish WRDS connection. Trying to load all data from cache if available."
        )
        if os.path.exists(initial_comp_cache_path):
            initial_df_from_wrds = pd.read_csv(
                initial_comp_cache_path, parse_dates=["datadate"]
            )
            print(
                f"Loaded initial Compustat data from cache: {initial_comp_cache_path}"
            )
        else:
            print(
                f"Cache file not found for initial Compustat data: {initial_comp_cache_path}. Cannot proceed."
            )
            initial_df_from_wrds = pd.DataFrame()  # Empty df

        if os.path.exists(adsprate_cache_path):
            comp_adsprate_df = pd.read_csv(
                adsprate_cache_path, parse_dates=["datadate"]
            )
            print(f"Loaded S&P ratings data from cache: {adsprate_cache_path}")
        else:
            print(
                f"Cache file not found for S&P ratings data: {adsprate_cache_path}. Proceeding without it."
            )
            comp_adsprate_df = pd.DataFrame()  # Empty df

    if initial_df_from_wrds is not None and not initial_df_from_wrds.empty:
        print("Starting main annual Compustat processing...")
        # The main_annual_compustat_processing function now takes the pre-joined dataframe
        # and the adsp_rate dataframe.
        final_processed_df = main_annual_compustat_processing(  # This call now matches the updated function
            initial_df_from_wrds,
            comp_adsprate_df,
        )

        print("\nFinal Processed DataFrame head:")
        print(final_processed_df.head())
        print("\nFinal Processed DataFrame tail:")
        print(final_processed_df.tail())
        print("\nFinal Processed DataFrame info:")
        final_processed_df.info()

        # Save the final processed DataFrame
        final_processed_df.to_csv(final_output_cache_path, index=False)
        print(f"Saved final processed data to: {final_output_cache_path}")
    else:
        print("Initial data is empty. Processing cannot continue.")
