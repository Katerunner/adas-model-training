import requests
import pandas as pd

from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor


def generate_basic_class_config_csv(config_path: str = 'configurations/tsd_dfg_classes.csv') -> pd.DataFrame:
    data = {
        'id': list(range(200)),
        'code': [
            'I-1', 'I-1.1', 'I-10', 'I-11', 'I-13', 'I-13.1', 'I-14', 'I-15', 'I-16', 'I-17', 'I-18', 'I-19', 'I-2',
            'I-2.1', 'I-20', 'I-25', 'I-27', 'I-28', 'I-28.1', 'I-29', 'I-29.1', 'I-3', 'I-30', 'I-32', 'I-34', 'I-36',
            'I-37', 'I-38', 'I-39-1', 'I-39-2', 'I-39-3', 'I-4', 'I-5', 'I-5.1', 'I-5.2', 'I-8', 'I-9', 'II-1',
            'II-10.1',
            'II-14', 'II-17', 'II-18', 'II-19-4', 'II-2', 'II-21', 'II-22', 'II-23', 'II-26', 'II-26.1', 'II-28',
            'II-3',
            'II-30-10', 'II-30-30', 'II-30-40', 'II-30-50', 'II-30-60', 'II-30-70', 'II-32', 'II-33', 'II-34', 'II-35',
            'II-39', 'II-4', 'II-40', 'II-41', 'II-42', 'II-42.1', 'II-43', 'II-45', 'II-45.1', 'II-45.2', 'II-46',
            'II-46.1', 'II-46.2', 'II-47', 'II-47.1', 'II-48', 'II-6', 'II-7', 'II-7.1', 'II-8', 'III-1', 'III-10',
            'III-105',
            'III-105.1', 'III-105.3', 'III-107-1', 'III-107-2', 'III-107.1-1', 'III-107.1-2', 'III-107.2-1',
            'III-107.2-2',
            'III-112', 'III-113', 'III-12', 'III-120', 'III-120-1', 'III-120.1', 'III-123', 'III-124', 'III-14',
            'III-14.1',
            'III-15', 'III-16', 'III-18-40', 'III-18-50', 'III-18-60', 'III-18-70', 'III-2', 'III-202-5', 'III-203-2',
            'III-206-1', 'III-21', 'III-23', 'III-25', 'III-25.1', 'III-27', 'III-29-30', 'III-29-40', 'III-3',
            'III-30-30',
            'III-33', 'III-34', 'III-35', 'III-37', 'III-39', 'III-40', 'III-42', 'III-43', 'III-45', 'III-46',
            'III-47',
            'III-5', 'III-50', 'III-54', 'III-59', 'III-6', 'III-64', 'III-68', 'III-74', 'III-77', 'III-78', 'III-8-1',
            'III-84', 'III-84-1', 'III-85-2', 'III-85-3', 'III-85.1', 'III-86-1', 'III-86-2', 'III-87', 'III-90',
            'III-90.1',
            'III-90.2', 'III-91', 'IV-1', 'IV-1.1', 'IV-10', 'IV-11', 'IV-12', 'IV-12.1', 'IV-13-1', 'IV-13-2',
            'IV-13-3',
            'IV-13-4', 'IV-13-5', 'IV-13-6', 'IV-13.1-2', 'IV-13.1-3', 'IV-13.1-4', 'IV-16', 'IV-17', 'IV-18', 'IV-2',
            'IV-20-1', 'IV-3-1', 'IV-3-2', 'IV-3-4', 'IV-3-5', 'IV-5', 'IV-6', 'VI-2.1', 'VI-3-1', 'VI-3-2', 'VI-3.1-1',
            'VI-3.1-2', 'VI-8', 'VII-4', 'VII-4-1', 'VII-4-2', 'VII-4.1-1', 'VII-4.3', 'VII-4.3-1', 'VII-4.3-2',
            'VII-4.4-1', 'VII-4.4-2', 'X-1.1', 'X-1.2', 'X-4', 'X-6-3'
        ],
        'name': ['' for _ in range(200)],
        'enabled': [1 for _ in range(200)]
    }

    df = pd.DataFrame(data)
    df.to_csv(config_path, index=False)
    return df


def _get_sign_name_from_wikipedia(sign_code):
    url = f'https://commons.wikimedia.org/wiki/File:Slovenia_road_sign_{sign_code}.svg'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    try:
        table = soup.find('table', class_="fileinfotpl-type-information vevent mw-content-ltr")
        cell = table.find('tr').find_all('td')[-1]
        div = cell.find('div')
        div.find('span').extract()
        return div.text.strip()
    except AttributeError:
        return ''


def populate_class_config_with_names(config_path: str = 'configurations/tsd_dfg_classes.csv'):
    df = pd.read_csv(config_path)
    codes = df['code'].tolist()

    with ThreadPoolExecutor(max_workers=200) as executor:
        names = list(executor.map(_get_sign_name_from_wikipedia, codes))

    df['name'] = names
    df.to_csv(config_path, index=False)
    return df
