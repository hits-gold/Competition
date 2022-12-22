#!/usr/bin/env python
# coding: utf-8

# In[4]:

import pandas as pd
import numpy as np
import gc
import os
import warnings
warnings.filterwarnings("ignore")

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import time
from datetime import datetime
import random
random.seed(1004)

class support():
    def __init__(self, data):
        self.data = data

        def make_basket(self):
            clacs_ = list(self.data.clac_mcls_nm.unique())
            basket = pd.DataFrame()
            for clac in clacs_:
                df = self.data.query("clac_mcls_nm == @clac & clac_hlv_nm != '담배'").groupby(
                    ['cust', "de_dt", "de_hr", "cop_c"]).pd_nm.apply(lambda x : list(x)).reset_index()
                df["pd_nm"] = df["pd_nm"].apply(lambda x : list(set(x)))
                df["cnt"] = df["pd_nm"].apply(lambda x : len(x))
                df = df.query("cnt > 1")
                df["clac_mcls_nm"] = clac
                basket = pd.concat([basket, df])
            return basket

        def make_support_score(self, basket):
            support = pd.DataFrame()
            for clac in basket.clac_mcls_nm.unique():
                lst = []
                for j in basket.query("clac_mcls_nm == @clac").pd_nm.values:
                    lst.append(j)
                te = TransactionEncoder()
                te_ary = te.fit_transform(lst)
                df = pd.DataFrame(te_ary, columns=te.columns_)
                frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
                frequent_itemsets["clac_mcls_nm"] = clac
                support = pd.concat([support, frequent_itemsets])
                support["cnt"] = support["itemsets"].apply(lambda x : len(list(x)))
                support = support.query("cnt > 1")
                support["itemsets"] = support["itemsets"].apply(lambda x : list(x))
            return support
        
        self.basket = make_basket(self)
        self.support = make_support_score(self, self.basket)
    
    def pro_recommend(self, product):
        lst_ = []
        for i in range(self.support.shape[0]):
            if product in self.support.iloc[i, 1]:
                lst_.append(i)
        pro_ = []
        for i in self.support.iloc[lst_, 1].values:
            pr = i.copy(); pr.remove(product)
            pro_.extend(pr)
        print(pro_, "를 추천드립니다")


# In[5]:


class ESG_cal():
    def __init__(self, score_data, cust_id):
        self.cust_id = cust_id
        self.score_data = score_data
        
    def lookup_score(self):
        if self.cust_id in self.score_data.cust.unique():
            df = self.score_data.query("cust == @self.cust_id").ESG_score.sum()
            print(f"현재 누적 ESG score는 {df}입니다.")
            return df
        else:
            print("누적된 ESG score가 없습니다.")
    
    def new_pay(self):
        input_info = ['온라인상품수', '거주지내상품수','저탄소상품액', '중고거래상품액', '친환경포장재상품액', 'ESG협력사상품액']
        info_dic = {}; info_dic["cust"] = self.cust_id
        info_dic["cop_c"] = input("cop_c :" )
        info_dic["de_dt"] = datetime.today().strftime("%Y-%m-%d")
        info_dic["de_hr"] = datetime.today().hour
        info_dic["chnl_dv"] = int(input("오프라인은 1, 온라인은 2 : "))
        
        input_info_dict = {}
        for input_data in input_info:
            input_info_dict[input_data] = int(input(f"{input_data} : "))
            
        info_dic["온라인상품수_마일리지"] = input_info_dict["온라인상품수"] * 700
        info_dic["거주지내구매_감점"] = input_info_dict["거주지내상품수"] * (-700)
        info_dic["상품액_마일리지"] = sum([input_info[pr] for pr in list(info_dic.keys()) if '액' in pr])
        info_dic["ESG_score"] = info_dic["온라인상품수_마일리지"] + info_dic["거주지내구매_감점"] + info_dic["상품액_마일리지"]
        self.score_data.iloc[self.score_data.shape[0]-1] = info_dic
        
        
        print("\n")
        print("이번 결제의 ESG score : ", self.score_data.loc[self.score_data.shape[0]-1, "ESG_score"])

# In[6]:


class collab_filtering():
    def __init__(self, model, items, pro_list, custID):
        self.model = model
        self.items = items
        self.pro_list = pro_list
        self.custID = custID
    
        def get_unbought(self, items, pro_list, custID):
            # 특정 custId가 구매내역이 있는 상품
            bought_item = items[items['custID']== custID]['item'].tolist()

            # 특정 custId가 구매내역이 없 상품
            unbought_item = [item for item in pro_list if item not in bought_item]

            # 전체 상품 수, 구매내역 있는 상품 수, 구매내역 없는 상품 수
            total_product_cnt = len(pro_list)
            bought_cnt = len(bought_item)
            unbought_cnt = len(unbought_item)

            print(f"전체 상품 수: {total_product_cnt}, 구매 내역있는 상품 수: {bought_cnt}, 추천 대상 상품 수: {unbought_cnt}")
            return unbought_item
        
        self.unbought_item = get_unbought(self, self.items, self.pro_list, self.custID)

    def recomm_item_by_surprise(self, top_n=10):

        # 아직 구매경험이 없은 상품에 대한 호감도 예측: prediction 객체 생성
        predictions = []    
        for item in self.unbought_item:
            predictions.append(self.model.predict(str(self.custID), str(item)))

        # 리스트 내의 prediction 객체의 est를 기준으로 내림차순 정렬
        def sortkey_est(pred):
            return pred.est

        predictions.sort(key=sortkey_est, reverse=True) # key에 리스트 내 객체의 정렬 기준을 입력

        # 상위 top_n개의 prediction 객체
        top_predictions = predictions[:top_n]

        # 상품, 예측 호감도 출력
        print(f"Top-{top_n} 추천 상품 리스트")

        for pred in top_predictions:
            item_nm = pred.iid
            item_ratings = pred.est
            print(f"{item_nm}: {item_ratings:.2f}")


# In[ ]:




