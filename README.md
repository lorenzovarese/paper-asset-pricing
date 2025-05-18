# paper-asset-pricing: P.A.P.E.R.
Platform for Asset Pricing Experiment and Research

```bash
.
├── data/                               # raw or parquet asset-pricing data
├── configs/                            # YAML specs for each workflow
│   ├── aggregate/                      # aggregate-dataset configurations
│   ├── experiment/                     # experiment (train/val/test) configs
│   └── portfolio/                      # portfolio backtest configs (VAL90, VAL95…)
├── LICENSE
├── pyproject.toml                      # project metadata & dependencies
├── README.md
├── ruff.toml
├── src/paperassetpricing/              # core library
│   ├── cli.py                          # entry-point (Typer)
│   ├── commands/                       # CLI commands
│   │   ├── aggregate.py
│   │   ├── experiment.py               # run rolling-window experiments
│   │   ├── generate_mock_data.py
│   │   ├── macro_extraction.py
│   │   ├── portfolio.py                # portfolio backtest command
│   │   └── split_dataset_by_date.py
│   ├── connectors/                     # data loaders (local CSV, WRDS…)
│   ├── etl/                            # aggregation pipelines & schemas
│   ├── helpers/                        # utilities (date alignment, etc.)
│   ├── metrics/                        # regression & portfolio metrics
│   ├── models/                         # modeling interfaces & implementations
│   │   ├── base_model.py
│   │   ├── linear_model.py
│   │   ├── neural_network_model.py     # your new NN with ReLU
│   │   └── model_registry.py
│   ├── pipelines/                      # custom DAGs or Prefect flows
│   ├── portfolios/                     # backtest & performance code
│   │   └── performance.py
│   └── settings.py
├── tests/                              # pytest suites for each module
│   ├── test_aggregator.py
│   ├── test_experiment.py
│   └── test_portfolio.py
└── uv.lock                             # lockfile for reproducible builds
```

# Data

This platform supports multiple data sources. Since our primary goal was to replicate Gu et al. (2020), data ingestion was designed around the local datasets that each user already possesses. This approach avoids license issues associated with directly retrieving data from sources that the user may not have access to, and it also prevents privilege‑related problems when accessing WRDS during feature extraction.

In practice, replication packages for published papers are often incomplete. For example, you might want to publish the features you have extracted (including their values and identifiers such as permco) on the web, but you must instruct the person reproducing the study to obtain additional columns—such as returns or volume—directly from a source (e.g. WRDS) to avoid violating the provider’s license.

## Example: Gu et al. replication (2020)

**Empirical Asset Pricing via Machine Learning**

### 94 Firm Characteristics

The authors used an adapted version of the original ![SAS program](https://drive.google.com/file/d/0BwwEXkCgXEdRQWZreUpKOHBXOUU/view?usp=sharing&resourcekey=0-1xjZ8fAc0sTybVC6RADDCA) from Green. et al (2016) that connects to various WRDS‑subscribed providers to extract all data required to construct 94 firm‑characteristic columns. The output of this feature‑extraction process is available here:

![Download `Empirical Data` from "Empirical Asset Pricing via Machine Learning"](https://dachxiu.chicagobooth.edu/)
(There is a 1.4 GB ZIP file containing a 3.57 GB datashare.csv with 4,117,301 lines.)

The resulting dataset has the following header:

```
permno,DATE,mvel1,beta,betasq,chmom,dolvol,idiovol,indmom,mom1m,mom6m,mom12m,mom36m,
pricedelay,turn,absacc,acc,age,agr,bm,bm_ia,cashdebt,cashpr,cfp,cfp_ia,chatoia,
chcsho,chempia,chinv,chpmia,convind,currat,depr,divi,divo,dy,egr,ep,gma,grcapx,
grltnoa,herf,hire,invest,lev,lgr,mve_ia,operprof,orgcap,pchcapx_ia,pchcurrat,
pchdepr,pchgm_pchsale,pchquick,pchsale_pchinvt,pchsale_pchrect,
pchsale_pchxsga,pchsaleinv,pctacc,ps,quick,rd,rd_mve,rd_sale,realestate,roic,
salecash,saleinv,salerec,secured,securedind,sgr,sin,sp,tang,tb,aeavol,cash,
chtx,cinvest,ear,nincr,roaq,roavol,roeq,rsup,stdacc,stdcf,ms,baspread,ill,
maxret,retvol,std_dolvol,std_turn,zerotrade,sic2
```

* **permno**: The CRSP's permno, a unique identifier at the share‑class level. Some firms issue multiple share classes (each with its own PERMNO). This field allows you to merge with other datasets or identify the company’s ticker.
* **DATE**: Month‑end date in YYYYMMDD format.
* **94 lagged firm characteristics**: See the Gu et al. appendix for full definitions.
* **sic2**: The first two digits of the Standard Industrial Classification code on DATE.

### Where are the returns?

As noted above, the authors extracted all firm characteristics themselves, but they did not include a plain returns column in the replication package. Returns are available directly from WRDS/CRSP. To obtain them, log in with your WRDS credentials and download the return series for each permno.

### Returns data from WRDS/CRSP

1. **Log in to WRDS**

   * Go to [https://wrds-web.wharton.upenn.edu/](https://wrds-www.wharton.upenn.edu/) and enter your WRDS username and password.

2. **Navigate to the CRSP Monthly Stock File**

   * From the WRDS landing page, click **“CRSP”** in the left‑hand menu.
   * Under “CRSP Products,” select **“Stock / Security File”** and click on **“Monthly Stock File”**.

3. **Define your sample**

   * In the “Date Range” fields, enter the start and end months matching your characteristics dataset (e.g. 1957‑01 through 2021‑12).
   * For “Identifier,” choose **PERMNO** to align with your `permno` column.
   * Check the box with **"Search the entire database"**

4. **Select variables**

   * Under “Select Variables,” check **RET** (monthly return).

5. **Select query output**
   * Output Format: comma-delimited text `csv`
   * Compression type: Uncompressed
   * Date Format: YYYYMMDD (e.g. 19840725)
   * Click **“Submit Form”** at the bottom of the page. WRDS will process your request.

6. **Download results**

   * When the query completes, click the **Download .csv Output** button.

7. **Merge with your characteristics data**

   * Use the platform to merge this data with the firm characteristics dataset. See the implementation details in [local_loader.py](connectors/local/local_loader.py) if needed but a configuration file will be supported.

# Troubleshooting

Mock data generation available in the [scripts](scripts) folder:
```bash
uv run python -m scripts.generate_mock_data
```