import urllib.request
from bs4 import BeautifulSoup  # parse html
import re  # regex
import csv
import os
import json
import pandas as pd
import urllib.request
import sklearn.externals
from sklearn.externals import joblib
from selenium.webdriver.chrome.service import Service
from underthesea import word_tokenize  # word_tokenize of lines
import numpy as np
import transformers as ppb  # load model BERT
from transformers import BertModel, BertTokenizer
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model._logistic import LogisticRegression
# scrap comment = selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time


# import requests

def load_url_selenium_shopee(url):
    # Selenium

    driver = webdriver.Chrome(service=Service(r'C:\\Users\\HAU\\Downloads\\chromedriver.exe'))

    print("Loading url=", url)
    driver.get(url)
    list_review = []
    # just craw 10 page
    x = 0
    while x < 20:
        try:
            # Get the review details here
            WebDriverWait(driver, 5).until(
                EC.visibility_of_all_elements_located((By.CSS_SELECTOR, "div.shopee-product-rating")))
        except:
            print('No has comment')
            break

        product_reviews = driver.find_elements(By.CSS_SELECTOR, "[class='shopee-product-rating']")
        print(product_reviews[0].text)
        # Get product review
        for product in product_reviews:
            review = product.find_element(By.CSS_SELECTOR, "[class='Em3Qhp']").text
            if (review != "" or review.strip()):
                print(review, "\n")
                list_review.append(review)
        # Check for button next-pagination-item have disable attribute then jump from loop else click on the next button

        if len(driver.find_elements(By.CSS_SELECTOR, "button.shopee-icon-button--right[enable]")) > 0:
            break;
        else:
            button_next = WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "button.shopee-icon-button--right")))
            driver.execute_script("arguments[0].click();", button_next)
            print("next page")
            time.sleep(5)
            x += 1
    driver.close()
    return list_review


def standardize_data(row):
    # remove stopword
    # Remove . ? , at index final
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Remove all . , " ... in sentences
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")

    row = row.strip()
    return row


# Tokenizer
def tokenizer(row):
    return word_tokenize(row, format="text")


def analyze(result):
    bad = np.count_nonzero(result)
    good = len(result) - bad
    print("No of bad and neutral comments = ", bad)
    print("No of good comments = ", good)

    if good > bad:
        return "Good! You can buy it!"
    else:
        return "Bad! Please check it carefully!"


def processing_data(data):
    # 1. Standardize data
    data_frame = pd.DataFrame(data)

    print('data frame:', data_frame)

    data_frame[0] = data_frame[0].apply(standardize_data)

    # 2. Tokenizer
    data_frame[0] = data_frame[0].apply(tokenizer)

    # 3. Embedding
    X_val = data_frame[0]

    return X_val


def load_pretrainModel(data):
    '''
    Load pretrain model/ tokenizers
    Return : features
    '''
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # encode lines
    tokenized = data.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    # get lenght max of tokenized
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    print('max len:', max_len)

    # if lenght of tokenized not equal max_len , so padding value 0
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    print('padded:', padded[1])
    print('len padded:', padded.shape)

    # get attention mask ( 0: not has word, 1: has word)
    attention_mask = np.where(padded == 0, 0, 1)
    print('attention mask:', attention_mask[1])

    # Convert input to tensor
    padded = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    # Load model
    with torch.no_grad():
        last_hidden_states = model(padded, attention_mask=attention_mask)
    #     print('last hidden states:', last_hidden_states)

    features = last_hidden_states[0][:, 0, :].numpy()
    print('features:', features)

    return features


def predict(url):
    # 1. Load URL and print comments
    if url == "":
        url = "https://shopee.vn/-M%C3%A3-ELMALL7-gi%E1%BA%A3m-7-%C4%91%C6%A1n-5TR-Tivi-TCL-4K-UHD-Android-9.0-43-inch-43T65-H%C3%A0ng-Ch%C3%ADnh-H%C3%A3ng-Mi%E1%BB%85n-ph%C3%AD-l%E1%BA%AFp-%C4%91%E1%BA%B7t-i.96527712.8731094240"
    data = load_url_selenium_shopee(url)
    #     data = load_url_selenium_tiki(url)
    data = processing_data(data)

    df = data
    df.to_csv("data.csv", index=False, header=False)
    # features = load_pretrainModel(data)
    #
    # # 2. Load weights
    # model = joblib.load('save_model.pkl')
    # # 3. Result
    # result = model.predict(features)
    # print(result)
    # print(analyze(result))


# predict(url ='https://tiki.vn/cho-sua-nham-cay-tai-sao-nhung-gi-ta-biet-ve-thanh-cong-co-khi-lai-sai-p17879295.html?spid=42131408')
# predict(url = 'https://www.lazada.vn/products/155-tang-khan-giay-bobby-don-1199k-lon-sua-bot-pediasure-ba-huong-vani-16kg-i150498393-s158167966.html?spm=a2o4n.home.flashSale.2.1e8a6afe5IZkzz&search=1&mp=1&c=fs&clickTrackInfo=rs%3A0.0%3Bfs_item_discount_price%3A939.000%3Bitem_id%3A150498393%3Bmt%3Ahot%3Bfs_item_sold_cnt%3A1944%3Babid%3A238030%3Bfs_item_price%3A1.158.300%3Bpvid%3A1872f7e2-4329-4ab3-8e99-bd30122b4914%3Bfs_min_price_l30d%3A939000.0%3Bdata_type%3Aflashsale%3Bfs_pvid%3A1872f7e2-4329-4ab3-8e99-bd30122b4914%3Btime%3A1652593697%3Bfs_biz_type%3Afs%3Bscm%3A1007.17760.238030.%3Bchannel_id%3A0000%3Bfs_item_discount%3A19%25%3Bcampaign_id%3A173202&scm=1007.17760.238030.0')
predict(url='https://shopee.vn/B%E1%BB%99-%C4%91%E1%BB%93-ng%E1%BB%A7-2-d%C3%A2y-thun-s%E1%BB%AFa-%C4%91%E1%BB%93-b%E1%BB%99-n%E1%BB%AF-m%E1%BA%B7c-nh%C3%A0-m%E1%BB%81m-m%C3%A1t-h%E1%BB%8Da-ti%E1%BA%BFt-d%E1%BB%85-th%C6%B0%C6%A1ng-BC11-i.240481126.7838624841?sp_atk=4bb370eb-dd9a-4919-91c5-b57343e5a787&xptdk=4bb370eb-dd9a-4919-91c5-b57343e5a787')
