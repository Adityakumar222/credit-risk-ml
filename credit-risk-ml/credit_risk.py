{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
import numpy as np\
\
from sklearn.model_selection import train_test_split\
from sklearn.preprocessing import StandardScaler\
from sklearn.metrics import classification_report, roc_auc_score\
from sklearn.linear_model import LogisticRegression\
from sklearn.ensemble import RandomForestClassifier\
\
df = pd.read_csv("loan_data.csv")\
\
df["Loan_Status"] = df["Loan_Status"].map(\{"Y": 1, "N": 0\})\
\
df = df.dropna()\
\
X = df.drop("Loan_Status", axis=1)\
y = df["Loan_Status"]\
\
X_train, X_test, y_train, y_test = train_test_split(\
    X, y, test_size=0.2, stratify=y, random_state=42\
)\
\
\
scaler = StandardScaler()\
X_train = scaler.fit_transform(X_train)\
X_test = scaler.transform(X_test)\
\
lr = LogisticRegression(max_iter=1000)\
lr.fit(X_train, y_train)\
\
lr_pred = lr.predict(X_test)\
lr_prob = lr.predict_proba(X_test)[:, 1]\
\
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, lr_prob))\
\
rf = RandomForestClassifier(\
    n_estimators=200,\
    max_depth=8,\
    random_state=42\
)\
rf.fit(X_train, y_train)\
\
rf_pred = rf.predict(X_test)\
rf_prob = rf.predict_proba(X_test)[:, 1]\
\
print("Random Forest ROC-AUC:", roc_auc_score(y_test, rf_prob))\
print(classification_report(y_test, rf_pred))\
}