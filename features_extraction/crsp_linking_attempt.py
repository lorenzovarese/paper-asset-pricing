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


def get_crsp_ccmxpf_linktable_from_wrds(
    conn: wrds.Connection, cache_path: str
) -> pd.DataFrame:
    """
    Fetches CRSP CCMXPF Linktable data directly from WRDS.
    Saves the fetched data to cache_path.
    """
    if conn is None:
        print(
            "WRDS connection is not available. Cannot fetch CRSP CCMXPF Linktable data."
        )
        # Optionally, try to load from cache as a fallback if conn is None
        if os.path.exists(cache_path):
            print(
                f"Loading CRSP CCMXPF Linktable data from cache as WRDS connection is unavailable: {cache_path}"
            )
            return pd.read_csv(cache_path, parse_dates=["linkdt", "linkenddt"])
        else:
            print(
                f"Cache file not found and WRDS connection unavailable for CRSP CCMXPF Linktable: {cache_path}"
            )
            return pd.DataFrame()

    print("Fetching CRSP CCMXPF Linktable data from WRDS...")
    query_linktable = """
    SELECT gvkey, lpermno, linktype, linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    """
    df = conn.raw_sql(query_linktable, date_cols=["linkdt", "linkenddt"])

    # Save to cache after fetching
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Saved CRSP CCMXPF Linktable data to cache: {cache_path}")
    except Exception as e:
        print(f"Error saving CRSP CCMXPF Linktable data to cache: {e}")

    return df


def get_crsp_mseall_from_wrds(conn: wrds.Connection, cache_path: str) -> pd.DataFrame:
    """
    Fetches CRSP MSEALL data directly from WRDS.
    Saves the fetched data to cache_path.
    """
    if conn is None:
        print("WRDS connection is not available. Cannot fetch CRSP MSEALL data.")
        # Optionally, try to load from cache as a fallback if conn is None
        if os.path.exists(cache_path):
            print(
                f"Loading CRSP MSEALL data from cache as WRDS connection is unavailable: {cache_path}"
            )
            return pd.read_csv(cache_path, parse_dates=["date"])
        else:
            print(
                f"Cache file not found and WRDS connection unavailable for CRSP MSEALL: {cache_path}"
            )
            return pd.DataFrame()

    print("Fetching CRSP MSEALL data from WRDS...")
    query_mseall = """
    SELECT date, permno, exchcd, shrcd, siccd
    FROM crsp.mseall 
    """
    df = conn.raw_sql(query_mseall, date_cols=["date"])

    # Save to cache after fetching
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Saved CRSP MSEALL data to cache: {cache_path}")
    except Exception as e:
        print(f"Error saving CRSP MSEALL data to cache: {e}")

    return df


def prepare_crsp_linktable(linktable_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the CRSP link table by filtering and sorting.
    Corresponds to:
    proc sort data=crsp.ccmxpf_linktable out=lnk;
      where LINKTYPE in ("LU", "LC", "LD", "LF", "LN", "LO", "LS", "LX") and
            (2015 >= year(LINKDT) or LINKDT = .B) and (1950 <= year(LINKENDDT) or LINKENDDT = .E);
      by GVKEY LINKDT; run;
    """
    df = linktable_df.copy()

    # Ensure date columns are datetime
    df["linkdt"] = pd.to_datetime(df["linkdt"], errors="coerce")
    df["linkenddt"] = pd.to_datetime(df["linkenddt"], errors="coerce")

    # Filter LINKTYPE
    valid_linktypes = ["LU", "LC", "LD", "LF", "LN", "LO", "LS", "LX"]
    df = df[df["linktype"].isin(valid_linktypes)]

    # Date filtering
    # (2015 >= year(LINKDT) or LINKDT = .B)
    cond_linkdt = (df["linkdt"].dt.year <= 2015) | df["linkdt"].isna()
    # (1950 <= year(LINKENDDT) or LINKENDDT = .E)
    cond_linkenddt = (df["linkenddt"].dt.year >= 1950) | df["linkenddt"].isna()

    df = df[cond_linkdt & cond_linkenddt]

    # Sort by GVKEY, LINKDT
    df = df.sort_values(by=["gvkey", "linkdt"])
    return df


def merge_compustat_with_linktable(
    compustat_df: pd.DataFrame, link_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges Compustat data with the prepared CRSP link table to get PERMNO.
    Corresponds to:
    proc sql; create table temp as select a.lpermno as permno,b.*
        from lnk a,data b where a.gvkey=b.gvkey
        and (LINKDT <= b.datadate or LINKDT = .B) and (b.datadate <= LINKENDDT or LINKENDDT = .E)
        and lpermno ne . and not missing(b.gvkey);
    quit;
    data temp;
        set temp;
        where not missing(permno);
    run;
    """
    # Ensure gvkey types are consistent for merging
    compustat_df["gvkey"] = compustat_df["gvkey"].astype(str)
    link_df["gvkey"] = link_df["gvkey"].astype(str)

    # Perform the merge (cross join then filter, or iterate if too large, but pandas merge handles it)
    # Using an inner merge on gvkey first
    merged_df = pd.merge(link_df, compustat_df, on="gvkey", how="inner")

    # Apply link date conditions
    # (LINKDT <= b.datadate or LINKDT = .B)
    cond_date_start = (merged_df["linkdt"] <= merged_df["datadate"]) | merged_df[
        "linkdt"
    ].isna()
    # (b.datadate <= LINKENDDT or LINKENDDT = .E)
    cond_date_end = (merged_df["datadate"] <= merged_df["linkenddt"]) | merged_df[
        "linkenddt"
    ].isna()

    merged_df = merged_df[cond_date_start & cond_date_end]

    # Filter for valid lpermno and gvkey (gvkey already handled by inner merge)
    merged_df = merged_df[merged_df["lpermno"].notna()]

    # Select columns: lpermno as permno, and all columns from compustat_df (which are already there with original names)
    merged_df = merged_df.rename(columns={"lpermno": "permno"})

    # Keep only columns from original compustat_df plus the new 'permno'
    # and essential linktable columns if needed for subsequent steps, or drop linktable specific ones.
    # SAS `b.*` implies keeping all columns from `data` (compustat_df).
    # We need to ensure we don't have duplicate columns from the merge if compustat_df also had linkdt etc.
    # Assuming compustat_df does not have linkdt, linkenddt, linktype.
    # The required columns are permno + original compustat_df columns.

    # Get original compustat columns
    comp_cols = list(compustat_df.columns)
    # Construct the list of columns to keep
    # Ensure 'permno' is present and gvkey (the join key)
    # If 'gvkey' was not in comp_cols (e.g. it was an index), add it.
    if "gvkey" not in comp_cols:
        final_cols = ["permno", "gvkey"] + [col for col in comp_cols if col != "gvkey"]
    else:
        final_cols = ["permno"] + comp_cols

    # Ensure no duplicate column names in final_cols before selecting
    # This can happen if 'permno' was already in comp_cols
    seen = set()
    unique_final_cols = []
    for x in final_cols:
        if x not in seen:
            seen.add(x)
            unique_final_cols.append(x)

    # Select the desired columns. If any column in unique_final_cols is not in merged_df, it will raise error.
    # This happens if compustat_df columns were dropped or renamed unexpectedly.
    # A safer way is to select 'permno' and all columns that originated from compustat_df

    # Let's select permno and all columns that were in compustat_df
    # All columns from compustat_df are already in merged_df from the `b.*` equivalent
    # We just added 'permno' (renamed from 'lpermno')
    # We need to drop the other columns from link_df: 'linktype', 'linkdt', 'linkenddt'
    # (unless 'lpermno' was already dropped by rename)

    cols_to_drop_from_link = ["linktype", "linkdt", "linkenddt"]
    # If lpermno was not dropped by rename (e.g. if it was copied), add it to drop list.
    # However, rename changes the name, so 'lpermno' as a column name should be gone.

    # Select all columns from the original compustat_df and the new permno
    # This is effectively what `select a.lpermno as permno, b.*` does.
    # Pandas merge already did this, we just need to ensure 'permno' is the correct name.
    # And filter out rows where permno is missing (already done by merged_df['lpermno'].notna())

    # The columns in merged_df are: gvkey, lpermno, linktype, linkdt, linkenddt (from link_df)
    # and all columns from compustat_df (including gvkey_y if gvkey was in both, but we merged on 'gvkey')
    # So, we need to select 'permno' (which is 'lpermno') and the compustat_df columns.

    # Let's refine column selection:
    output_df = merged_df[["permno"] + comp_cols]

    # Final filter: where not missing(permno)
    # This was already implicitly handled by merged_df['lpermno'].notna() before rename
    # If permno could become NA after rename (not possible here), then filter again:
    output_df = output_df[output_df["permno"].notna()].reset_index(drop=True)

    return output_df


def prepare_crsp_mseall(mseall_df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares CRSP MSEALL data by filtering, calculating exchange date ranges, and sorting.
    Corresponds to:
    proc sort data=crsp.mseall(keep=date permno exchcd shrcd siccd) out=mseall nodupkey;
        where exchcd in (1,2,3) or shrcd in (10,11,12);
        by permno exchcd date; run;
    proc sql; create table mseall as
        select *,min(date) as exchstdt,max(date) as exchedt
        from mseall group by permno,exchcd; quit;
    proc sort data=mseall nodupkey;
        by permno exchcd; run;
    """
    # Columns are already selected if get_crsp_mseall_from_wrds is used
    df = mseall_df_raw[["date", "permno", "exchcd", "shrcd", "siccd"]].copy()
    df["date"] = pd.to_datetime(df["date"])

    # Filter: where exchcd in (1,2,3) or shrcd in (10,11,12);
    cond_exchcd = df["exchcd"].isin([1, 2, 3])
    cond_shrcd = df["shrcd"].isin([10, 11, 12])
    df_filtered = df[cond_exchcd | cond_shrcd]

    # Sort and drop duplicates: by permno exchcd date; nodupkey;
    df_sorted = df_filtered.sort_values(by=["permno", "exchcd", "date"])
    # nodupkey in SAS proc sort means drop duplicates for the 'by' variables.
    # Here, the by variables are permno, exchcd, date.
    df_deduped = df_sorted.drop_duplicates(
        subset=["permno", "exchcd", "date"], keep="first"
    )

    # Calculate exchstdt and exchedt
    # select *,min(date) as exchstdt,max(date) as exchedt from mseall group by permno,exchcd;
    # Pandas transform can add these back to the original shape, or agg and merge.
    # SAS proc sql does a group by, so the output rows are unique by permno, exchcd.
    # All other columns (shrcd, siccd, date) in the select * will be from one of the rows in the group.
    # SAS SQL typically takes the first or last row's values for non-aggregated, non-group-by columns.
    # To be safe, let's aggregate and then merge back if we need other columns, or just keep aggregated.
    # The SAS code implies that shrcd, siccd are carried over. Let's take the first.

    df_agg = df_deduped.groupby(["permno", "exchcd"], as_index=False).agg(
        exchstdt=("date", "min"),
        exchedt=("date", "max"),
        # Carry over other columns by taking the first value in the group
        # This matches SAS behavior for non-aggregate columns in a grouped query's select list.
        shrcd=("shrcd", "first"),
        siccd=("siccd", "first"),
        # The original 'date' column itself is not explicitly carried over in SAS SQL this way,
        # but exchstdt/exchedt are derived from it.
        # The later join uses exchstdt and exchedt.
    )

    # Sort again: by permno exchcd; nodupkey;
    # The groupby already makes permno, exchcd unique. Sorting is for order.
    df_final_mseall = df_agg.sort_values(by=["permno", "exchcd"])
    # nodupkey is implicit due to groupby

    return df_final_mseall


def screen_with_market_info(
    temp_df: pd.DataFrame, mseall_prepared_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins the temporary data with prepared MSEALL data and applies final screens.
    Corresponds to:
    proc sql; create table temp as select *
        from temp as a left join mseall as b
        on a.permno=b.permno
        and exchstdt<=datadate<= exchedt;
    quit;
    data temp;
        set temp;
           where exchcd in (1,2,3) and shrcd in (10,11) and not missing(permno);
        drop shrcd date siccd exchstdt exchedt;
    run;
    proc sort data=temp nodupkey;
        by gvkey datadate;
    run;
    """
    # Ensure permno types are consistent for merging
    # permno in temp_df could be float if from lpermno, ensure it's compatible with mseall_prepared_df.permno
    temp_df["permno"] = pd.to_numeric(
        temp_df["permno"], errors="coerce"
    )  # lpermno is numeric
    mseall_prepared_df["permno"] = pd.to_numeric(
        mseall_prepared_df["permno"], errors="coerce"
    )

    # Left join: on a.permno=b.permno
    # The date condition exchstdt<=datadate<= exchedt needs to be applied post-merge.
    merged_df = pd.merge(temp_df, mseall_prepared_df, on="permno", how="left")

    # Apply date condition: and exchstdt<=datadate<= exchedt;
    # Ensure datadate is datetime
    merged_df["datadate"] = pd.to_datetime(merged_df["datadate"])
    # exchstdt and exchedt should be datetime from prepare_crsp_mseall

    cond_date_range = (merged_df["exchstdt"] <= merged_df["datadate"]) & (
        merged_df["datadate"] <= merged_df["exchedt"]
    )

    # For rows where the join didn't find a match in mseall, exchstdt/exchedt will be NaN.
    # The condition exchstdt <= datadate <= exchedt will be false for these NaNs.
    # SAS join condition like this effectively filters out non-matches or where condition is false.
    merged_df_filtered_date = merged_df[cond_date_range]

    # Filter: where exchcd in (1,2,3) and shrcd in (10,11) and not missing(permno);
    cond_final_exchcd = merged_df_filtered_date["exchcd"].isin([1, 2, 3])
    cond_final_shrcd = merged_df_filtered_date["shrcd"].isin(
        [10, 11]
    )  # Note: (10,11) not (10,11,12)
    cond_permno_ok = merged_df_filtered_date["permno"].notna()

    final_df = merged_df_filtered_date[
        cond_final_exchcd & cond_final_shrcd & cond_permno_ok
    ]

    # Drop columns: drop shrcd date siccd exchstdt exchedt;
    # 'date' here refers to the date column from mseall, which was used to create exchstdt/exchedt.
    # mseall_prepared_df contains exchstdt, exchedt, shrcd, siccd.
    # It does not contain the original 'date' column from crsp.mseall after the groupby.
    # So, we drop the columns that were brought in from mseall_prepared_df and are no longer needed.
    cols_to_drop = ["shrcd", "siccd", "exchstdt", "exchedt"]
    # If 'date' from mseall was somehow carried into mseall_prepared_df and then merged, it should be dropped.
    # Based on prepare_crsp_mseall, 'date' is not in mseall_prepared_df.
    # exchcd is used in the where clause but not dropped. It will be kept.

    final_df = final_df.drop(columns=cols_to_drop, errors="ignore")

    # Sort: by gvkey datadate; nodupkey;
    final_df_sorted = final_df.sort_values(by=["gvkey", "datadate"])
    final_df_deduped = final_df_sorted.drop_duplicates(
        subset=["gvkey", "datadate"], keep="first"
    )

    return final_df_deduped.reset_index(drop=True)


def process_merge_with_crsp(
    compustat_annual_df: pd.DataFrame,
    crsp_linktable_df_raw: pd.DataFrame,
    crsp_mseall_df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Orchestrates the merging of Compustat annual data with CRSP link table and MSEALL.
    """
    print("Step 8: Preparing CRSP link table...")
    link_df_prepared = prepare_crsp_linktable(crsp_linktable_df_raw)
    print(f"Link table prepared. Shape: {link_df_prepared.shape}")

    print("Step 9: Merging Compustat data with prepared CRSP link table...")
    # compustat_annual_df is 'data' in SAS, becomes 'temp' after this
    temp_df_with_permno = merge_compustat_with_linktable(
        compustat_annual_df, link_df_prepared
    )
    print(f"Merged with link table. Shape: {temp_df_with_permno.shape}")
    if "permno" not in temp_df_with_permno.columns:
        print("Warning: 'permno' column not found after merging with link table.")
        # Handle error or return empty/original df
        return temp_df_with_permno
    if temp_df_with_permno["permno"].isna().all():
        print("Warning: All 'permno' values are NaN after merging with link table.")

    print("Step 10: Preparing CRSP MSEALL data...")
    mseall_prepared = prepare_crsp_mseall(crsp_mseall_df_raw)
    print(f"CRSP MSEALL data prepared. Shape: {mseall_prepared.shape}")

    print("Step 11: Screening with CRSP market information (MSEALL)...")
    # temp_df_with_permno is 'temp' before joining with mseall
    final_temp_df = screen_with_market_info(temp_df_with_permno, mseall_prepared)
    print(
        f"Screened with market info. Final shape for this stage: {final_temp_df.shape}"
    )

    return final_temp_df


if __name__ == "__main__":
    # Define cache paths
    base_cache_dir = "cache/data"
    initial_comp_cache_path = os.path.join(
        base_cache_dir, "compustat_annual_initial_merged.csv"
    )
    adsprate_cache_path = os.path.join(base_cache_dir, "compustat_adsprate.csv")

    # New cache paths for CRSP data
    crsp_linktable_cache_path = os.path.join(
        base_cache_dir, "crsp_ccmxpf_linktable.csv"
    )
    crsp_mseall_cache_path = os.path.join(base_cache_dir, "crsp_mseall.csv")

    final_output_after_crsp_merge_cache_path = os.path.join(
        base_cache_dir, "compustat_annual_crsp_merged_features.csv"
    )
    # Path for saving output if CRSP processing is not done (e.g. if we revert to skipping)
    # For now, with ValueError, this specific path might not be used if CRSP data is mandatory.
    # final_output_wo_crsp_cache_path = os.path.join(
    #     base_cache_dir, "compustat_annual_processed_without_crsp.csv"
    # )

    os.makedirs(base_cache_dir, exist_ok=True)

    print("Attempting to open WRDS connection...")
    wrds_conn = open_wrds_connection()

    initial_df_from_wrds = None
    comp_adsprate_df = None
    # Initialize CRSP dataframes
    crsp_ccmxpf_linktable_raw_df = pd.DataFrame()
    crsp_mseall_raw_df = pd.DataFrame()

    if wrds_conn:
        try:
            # Fetch Compustat data
            initial_df_from_wrds = get_compustat_initial_data_from_wrds(
                wrds_conn, initial_comp_cache_path
            )
            comp_adsprate_df = get_adsprate_data_from_wrds(
                wrds_conn, adsprate_cache_path
            )

            # Fetch CRSP data directly from WRDS
            print("\nFetching CRSP data from WRDS...")
            crsp_ccmxpf_linktable_raw_df = get_crsp_ccmxpf_linktable_from_wrds(
                wrds_conn, crsp_linktable_cache_path
            )
            crsp_mseall_raw_df = get_crsp_mseall_from_wrds(
                wrds_conn, crsp_mseall_cache_path
            )

        except Exception as e:
            print(f"An error occurred during data fetching from WRDS: {e}")
            # CRSP dataframes will remain empty if an error occurs here
        finally:
            print("Closing WRDS connection.")
            wrds_conn.close()
    else:
        print(
            "Could not establish WRDS connection. Trying to load ALL data from cache if available."
        )
        # Fallback to cache for Compustat data
        if os.path.exists(initial_comp_cache_path):
            initial_df_from_wrds = pd.read_csv(
                initial_comp_cache_path, parse_dates=["datadate"]
            )
            print(
                f"Loaded initial Compustat data from cache: {initial_comp_cache_path}"
            )
        else:
            print(
                f"Cache file not found for initial Compustat data: {initial_comp_cache_path}."
            )
            initial_df_from_wrds = pd.DataFrame()

        if os.path.exists(adsprate_cache_path):
            comp_adsprate_df = pd.read_csv(
                adsprate_cache_path, parse_dates=["datadate"]
            )
            print(f"Loaded S&P ratings data from cache: {adsprate_cache_path}")
        else:
            print(f"Cache file not found for S&P ratings data: {adsprate_cache_path}.")
            comp_adsprate_df = pd.DataFrame()

        # Fallback to cache for CRSP data since WRDS connection failed
        if os.path.exists(crsp_linktable_cache_path):
            crsp_ccmxpf_linktable_raw_df = pd.read_csv(
                crsp_linktable_cache_path, parse_dates=["linkdt", "linkenddt"]
            )
            print(f"Loaded CRSP Linktable data from cache: {crsp_linktable_cache_path}")
        else:
            print(
                f"Cache file not found for CRSP Linktable: {crsp_linktable_cache_path}."
            )
            # crsp_ccmxpf_linktable_raw_df remains pd.DataFrame()

        if os.path.exists(crsp_mseall_cache_path):
            crsp_mseall_raw_df = pd.read_csv(
                crsp_mseall_cache_path, parse_dates=["date"]
            )
            print(f"Loaded CRSP MSEALL data from cache: {crsp_mseall_cache_path}")
        else:
            print(f"Cache file not found for CRSP MSEALL: {crsp_mseall_cache_path}.")
            # crsp_mseall_raw_df remains pd.DataFrame()

    if initial_df_from_wrds is not None and not initial_df_from_wrds.empty:
        print("Starting main annual Compustat processing...")
        df_annual_compustat_processed = main_annual_compustat_processing(
            initial_df_from_wrds,
            comp_adsprate_df,
        )
        print("\nAnnual Compustat Processing DataFrame head:")
        print(df_annual_compustat_processed.head())

        # Check if CRSP data was successfully fetched or loaded
        if (
            crsp_ccmxpf_linktable_raw_df is not None
            and not crsp_ccmxpf_linktable_raw_df.empty
            and crsp_mseall_raw_df is not None
            and not crsp_mseall_raw_df.empty
        ):
            print("\nStarting merge with CRSP process...")
            df_after_crsp_merge = process_merge_with_crsp(
                df_annual_compustat_processed,
                crsp_ccmxpf_linktable_raw_df,
                crsp_mseall_raw_df,
            )

            print("\nDataFrame head after CRSP merge and screen:")
            print(df_after_crsp_merge.head())
            print("\nDataFrame tail after CRSP merge and screen:")
            print(df_after_crsp_merge.tail())
            print("\nDataFrame info after CRSP merge and screen:")
            df_after_crsp_merge.info(verbose=True, show_counts=True)

            df_after_crsp_merge.to_csv(
                final_output_after_crsp_merge_cache_path, index=False
            )
            print(
                f"Saved data after CRSP merge to: {final_output_after_crsp_merge_cache_path}"
            )
        else:
            # Instead of skipping, raise an error to halt execution.
            # The print statements from the data fetching attempts (WRDS or cache)
            # should provide context as to why the data might be missing.
            raise ValueError(
                "CRSP link table or MSEALL data is missing or empty. "
                "Cannot proceed with CRSP-dependent steps. "
                "Please check WRDS connection, SQL queries (e.g., table names 'crsp.ccmxpf_linktable', 'crsp.mseall'), "
                "data availability in WRDS, or local cache files if applicable."
            )
    else:
        print("Initial Compustat data is empty. Processing cannot continue.")
