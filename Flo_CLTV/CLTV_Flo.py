##########################################
# CLTV Prediction with BG-NBD and Gamma-Gamma
##########################################

"""  MAIN PURPOSE:  Roadmap for FLO sales and marketing activities
wants to determine. The company's medium-long-term plan
existing customers to the company in the future so that they can
estimating the potential value they will provide
required.
"""

# master id: Unique number of Customers
# order_chanel: Which channel of the shopping platform is used (Android, ios, Desktop, Mobile)
# last order channel: The channel where the last order was made
# first_order_date: Customer's first order date
# last_order_date: Customer's last order date
# last_order_date_online: Customer's last order date at Online platform
# last_order_date_offline: Customer's last order date at Offline platform
# order_num_total_ever_online: The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline: The total number of purchases made by the customer on the offline platform
# customer_value_total_ever_offline: The total amount of purchases made by the customer on the offline platform
# customer_value_total_ever_online: The total amount of purchases made by the customer on the online platform
# interested_in_categories_12: List of categories the customer has shopped in the last 12 months

##############################################
# Step 1: Preparing the Data
##############################################

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
df_ = pd.read_csv("Flo_CLTV/flo_data_20k.csv")
df = df_.copy()

""" def missing_values_analysis(df):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=True)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df """


def check_dataframe(df, row_num=10):
    print("########## Dataset Shape ##########")
    print("Rows:", df.shape[0], "\nColumns:", df.shape[1])
    print("########## Dataset Information ##########")
    print(df.info())
    print("########## Types of Columns  ##########")
    print(df.dtypes)
    print("########## First {row_num} Rows ##########")
    print(df.head(row_num))
    print("########## Last{row_num} Rows ##########")
    print(df.tail(row_num))
    print("########## Summary Statistics of the Dataset ##########")
    print(df.describe().T)
    print("########## Nu. of Null Values ##########")
    print(df.isnull().sum())


check_dataframe(df)


# Step 1:
# Define the outlier thresholds and replace with thresholds functions needed to suppress outliers.
# ..Note: When calculating cltv, frequency values must be integers.
# .. Therefore, round the lower and upper limits with round() function.


def outlier_thresholds(df, col_name, q1=0.05, q3=0.95):
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "customer_value_total_ever_online", q1=0.01, q3=0.99)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

# Step 2: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# .."customer_value_total_ever_online" If the variables have outliers, suppress them.

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]

for i in num_cols:
    replace_with_thresholds(df, i)

# Step 3: Omnichannel means that customers shop from both online and offline platforms.
# .. Create new variables for each customer's total purchases and spending

df["total_order"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Step 4: Examine the variables types. Change the type of variables that express date to "datetime".

# date_columns = df.columns[df.columns.str.contains("date")]
date_columns = [col for col in df.columns if 'date' in col]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

##############################################
# Step 2: Creating the CLTV Data Structure
##############################################

# Step 2: Take 2 days after the date of the last purchase in the data set as the date of analysis.

df["last_order_date"].max()
last_date = dt.datetime(2021, 6, 1)

# Step 1: Create a new cltv Dataframe that includes customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg values.
# ..Monetary value will be expressed as average value per purchase, recency and tenure values ​​will be expressed in weekly terms.

# recency: How recently a customer has made a purchase.(For Every Customer)
# Tenure: Customer's order age. Weekly. (how long before the analysis date the first purchase was made)
# frequency: How often a customer makes a purchase (frequency>1)
# monetary: How much money a customer spends on purchases.


cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((last_date - df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["frequency"] = df["total_order"]
cltv_df["monetary_cltv_avg"] = df["total_customer_value"] / df["total_order"]

cltv_df.head()

##############################################
# Step 3: Görev 3: BG/NBD(Forecast of Expected Sales),
#                  Gamma-Gamma(Forecast of Expected Profitability)
#                  Establishment of Models and Calculation of CLTV
##############################################

# Step1: Please fit BG/NBD model:

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# Estimate the expected purchases from customers within 3 months and add the exp_sales_3_month column to the cltv dataframe.

cltv_df["exp_sales_3_month"] = bgf.predict()

# Estimate the expected purchases from customers within 6 months and add the exp_sales_3_month column to the cltv dataframe.

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])
cltv_df.head()

# Step 2: Fit the Gamma-Gamma model. Estimate the average value that customers will add and cltv as exp_average_value
# add to dataframe

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])

cltv_df.head()

# Step3: Calculate 6 months CLTV and add it to the dataframe with the name "cltv".
# Observe the 20 people with the highest cltv value

cltv_df['cltv'] = ggf.customer_lifetime_value(bgf,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'],
                                              cltv_df['monetary_cltv_avg'],
                                              time=6,
                                              freq="W",
                                              discount_rate=0.01)

cltv_df.sort_values("cltv", ascending=False).head(20)

##############################################
# Step 4: Creating Segments by CLTV value
##############################################

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

cltv_df.groupby("cltv_segment").agg({"count", "mean", "sum"})




