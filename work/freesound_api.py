
import requests
from bs4 import BeautifulSoup
from pprint import pprint
import json

import os
import math
import pathlib
from tqdm import tqdm
from humanfriendly import format_size, parse_size


token = 'dedOJD0nnsIebO5RcNQHqt9fLMElgDWccZi5rqcn'

def get_ids_freesound(query, page_num=1, page_size=15, save=True):
    global file_ids
    
    query = query
    page_num = page_num
    page_size = page_size
    
    print("Query: {}\nStarting page: {}\nPage size: {}\nSave: {}"
          .format(query, page_num, page_size, save))

    file_ids = []
    end_reached = False

    while not end_reached:
        res = requests.get('https://freesound.org/apiv2/search/text/',
                          params={'token' : token,
                                  'query' : query,
                                  'page' : page_num,
                                  'page_size' : page_size,
                                 })
        try:
            json_res = json.loads(res.content.decode('utf8'))
            if page_num == 1:
                print(json_res['count'], "results")
                print("Expected number of pages: {}"
                      .format(math.ceil(json_res['count']/page_size) - page_num + 1))
            file_ids.extend([res['id'] for res in json_res['results']])
            print(page_num, end=' ')
        except:
            end_reached = True
        page_num += 1
        
    if save:
        try:
            with open('../_data/{}_ids.txt'.format(query), 'x') as out_text:
                print("\nSaving to {}".format('../_data/{}_ids.txt'.format(query)))
                for item in file_ids:
                    out_text.write("{}\n".format(item))
        except FileExistsError as e:
                print("File already exists, skipping save.")
    
    return file_ids


def load_ids(query):
    with open('../_data/{}_ids.txt'.format(query), 'r+') as read_text:
        id_list = read_text.read().splitlines()
    return id_list


def save_mp3s(id_list, name='unknown', start_enum=1, size_threshold='100MB', overwrite=False):
    # First result has enum value of 1, instead of 0
    enum = start_enum
    
    print("Query: {}".format(name)) 
    pathlib.Path('../_data/mp3s/{}'.format(name)).mkdir(parents=True, exist_ok=True) 

    for file_id in id_list:
        file_path = "../_data/mp3s/{0}/{2:04}_{0}_{1}.mp3".format(name, file_id, enum)
        
        print("======== Result no: {} | File id: {} ========".format(enum, file_id))
        print("Saving to {}".format(file_path))
        
        if overwrite:
            mode = "wb+"
        else:
            mode = "xb"
        
        try:
            with open(file_path, mode) as handle:
                res = requests.get('https://freesound.org/apiv2/sounds/{}/'.format(file_id),
                                   params={'token' : token,
                                           })
                json_res = json.loads(res.content.decode('utf8'))
                filesize = json_res['filesize']
                print("File size is {}".format(format_size(filesize)))
                
                if filesize > parse_size(size_threshold):
                    raise Exception("size_threshold")
                
                url = json_res['previews']['preview-hq-mp3']
                response = requests.get(url, stream=True)

                for data in tqdm(response.iter_content()):
                    handle.write(data)
        except FileExistsError as e:
            print("File already exists, skipping.")
        except Exception as e:
            if str(e) != "size_threshold":
                raise
            else:
                print("File size greater than {}, skipping."
                     .format(format_size(parse_size(size_threshold))))
                os.remove(file_path)
        
        enum += 1
    return
        
        
def fetch_mp3s(query, startindex=0, endindex='', size_threshold='100MB', overwrite=False):
    try:
        query_all = load_ids(query)
    except:
        print("ID list not found")
        return    
    query_list = query_all[startindex:endindex]
    save_mp3s(query_list, query, start_enum=startindex+1, size_threshold=size_threshold,
              overwrite=overwrite)
    return