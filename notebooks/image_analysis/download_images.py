import os
import requests

from multiprocessing.pool import ThreadPool

f = open('test.csv')

locs = []
for l in f.readlines():
    ls = l.split(',')
    nm = ls[0].replace('"', '')
    nm = 'out/' + nm + '.jpg'
    loc = ls[1].replace('"', '')
    #loc = loc.strip()
    locs.append( (nm, loc) ) 

print(len(locs))

def fetch_url(entry):
    path, uri = entry
    try:
        if not os.path.exists(path):
            r = requests.get(uri, stream=True)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    for chunk in r:
                        f.write(chunk)
            print( path )
    except:
        path = "None"
    return path

results = ThreadPool(8).imap_unordered(fetch_url, locs)

print( list(results) )


