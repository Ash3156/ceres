from urllib.request import urlretrieve
import os
import csv
import glob

def download_images(url_dict, path):
    files = glob.glob(os.path.join(path, '*'))
    for f in files:
        os.remove(f)
    for key in url_dict.keys():
        fullfilename = os.path.join(path, key+".jpg")
        urlretrieve(url_dict[key]['Image URL'], fullfilename)
        
def create_dict(filename):
    plant_disease_dict = dict()
    diseases = []
    with open(filename, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count=0
        for row in csv_reader:
            if line_count==0:
                categories = row[1:]
                line_count+=1
            else:
                diseases.append(row[0])
                plant_disease_dict[row[0]]=dict()
                for i in range(len(categories)):
                    if row[i+1]:
                        plant_disease_dict[row[0]][categories[i]]=row[i+1]
    return (plant_disease_dict, diseases)

def create_link_dict(filename):
    link_dict = dict()
    with open(filename, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count=0
        for row in csv_reader:
            if line_count==0:
                line_count+=1
            else:
                link_dict[row[0]]=[]
                for i in range(len(row)-1):
                    if row[i+1]:
                        link_dict[row[0]].append(row[i+1])
    return link_dict