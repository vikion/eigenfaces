import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image

URLS = [
    "https://fmph.uniba.sk/pracoviska/katedra-didaktiky-matematiky-fyziky-a-informatiky/",
    "https://fmph.uniba.sk/pracoviska/katedra-jazykovej-pripravy/",
    "https://fmph.uniba.sk/pracoviska/katedra-telesnej-vychovy-a-sportu/",
    "https://fmph.uniba.sk/pracoviska/detasovane-pracovisko-turany/"
]

DUMMY = list(Image.open('fedora.jpg').getdata())
DIRN_PHOTOS = './photos/'
DIRN_DUMMIES = './dummies/'

photo_details = {}
dummies_details = {}

for dep in URLS:
    text = requests.get(dep).text
    parsed = BeautifulSoup(text, 'html.parser')
    employees = [href for href in parsed.select('article.span6 a') if '/ludia/' in href['href']]
    department = dep.split('/')[-2]
    for employee in employees:
        photo_url = employee['href'].replace('/ludia/', '/f/')
        username = photo_url.split('/')[-1]
        filename = f"{username}.jpg"
        name = employee.text

        im_list = list(Image.open(requests.get(photo_url, stream=True).raw).getdata())
        if im_list != DUMMY:
            dirn = DIRN_PHOTOS
            photo_details[filename] = [name, department]
        else:
            dirn = DIRN_DUMMIES
            dummies_details[filename] = [name, department]

        photo = requests.get(photo_url).content
        path = dirn + filename
        with open(path, 'wb') as handler:
            handler.write(photo)


df_photos = pd.DataFrame.from_dict(photo_details, orient='index', columns=['employee', 'department'])
df_photos.index.name = 'filename'
#df_photos.to_csv("med_photos.csv")

df_dummies = pd.DataFrame.from_dict(dummies_details, orient='index', columns=['employee', 'department'])
df_dummies.index.name = 'filename'
#df_dummies.to_csv("med_dummies.csv")
