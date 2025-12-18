convert and save Fraud_Data.csv as parquet file

```bash
df = pd.read_csv("Fraud_Data.csv")
df.to_parquet("Fraud_Data.parquet")
```

OR 

```bash
import pyarrow.csv aas pv
import pyarrow.parquet as pq

table = pv.read_csv("Fraud_Data.csv")
pq.write_table(table, "Fraud_Data.parquet")
```