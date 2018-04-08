# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import re
import time


def fetch():
    for i in range(50):
        form_data = {
            "OrderByField": "",
            "OrderByDesc": "",
            "currentPage": i + 50,
            "pageSize": 10
        }
        r = requests.post("http://www.bjjs.gov.cn/eportal/ui?pageId=303477", data=form_data)
        html = r.content.decode('utf-8')
        # with open("C:\\Users\\cashbet\\Desktop\\test.txt", encoding="utf-8") as f:
        #     html = f.read()

        # print(html)
        parse(html)
        time.sleep(3)


def parse(html):
    soup = BeautifulSoup(html)
    for elem in soup.find_all(name="div", attrs={"class": "md-content"}):
        temp = elem.find_all(name="li")[1]
        text = temp.text.strip()
        new_string = re.sub(re.compile('\s+'), '', text)
        print(new_string.strip())


if __name__ == '__main__':
    fetch()
