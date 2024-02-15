# %%
import numpy as np
import pandas as pd

train = pd.read_csv("../../input/open_v2/train.csv")
test = pd.read_csv("../../input/open_v2/test.csv")

# %%
train["disease_type"].value_counts()
# %%
test["disease_type"].value_counts()
# %%
train["disease_state"].unique()
# %%
test["disease_state"].value_counts()
# %%
set(train["disease_type"].unique()) - set(test["disease_type"].unique())
# %%
set(test["disease_type"].unique()) - set(train["disease_type"].unique())
# %%
set(test["disease_state"].unique()) - set(train["disease_state"].unique())
# %%
set(train["disease_state"].unique()) - set(test["disease_state"].unique())

# %%
set(train["reference_journal"].unique()) - set(test["reference_journal"].unique())
# %%
set(test["reference_journal"].unique()) - set(train["reference_journal"].unique())
# %%

train["assay_method_technique"]
# %%
test["assay_method_technique"]
# %%
set(train["assay_method_technique"].unique()) - set(test["assay_method_technique"].unique())
# %%
set(test["assay_method_technique"].unique()) - set(train["assay_method_technique"].unique())
# %%
features = [
    "assay_method_technique",
    "assay_group",
    "disease_type",
    "disease_state",
    "reference_date",
    "reference_journal",
    "reference_title",
]

train = pd.get_dummies(train, columns=features)
test = pd.get_dummies(test, columns=features)

train.shape
# %%
test.shape
# %%
