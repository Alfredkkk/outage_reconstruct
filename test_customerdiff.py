import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("~/scratch/data/grower/mandv/raw_data/ca/layout_investor/per_outage_investor_owned_2.csv")
    cts = df["ImpactedCustomers"]
    # group by OBJECTID
    cts = cts.groupby(df["OBJECTID"])
    for name, group in cts:
        if (group > 10000).sum() > 0:
            print(name, (group > 1000).sum())