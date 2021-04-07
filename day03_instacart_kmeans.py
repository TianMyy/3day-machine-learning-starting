import pandas as pd
from sklearn.decomposition import PCA

# get data after PCA dimension reduction
order_products = pd.read_csv("./instacart/order_products_prior.csv")
products = pd.read_csv("./instacart/products.csv")
orders = pd.read_csv("./instacart/order.csv")
aisles = pd.read_csv("./instacart/aisle.csv")
table1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
table2 = pd.merge(table1, order_products, on=["product_id", "product_id"])
table3 = pd.merge(table2, orders, on=["order_id", "order_id"])
table = pd.crosstab(table3["user_id"], table3["aisle"])
data = table[:10000]
transfer = PCA(n_components=0.95)
data_new = transfer.fit_transform(data)
# 2.combine files
# combine aisles and products

table3.head()

# 3.find the relation between user_id and aisle


# 4.PCA dimension reduction

