import pandas as pd
import numpy as np
from tqdm import tqdm

def get_category2id():
    file = "E:\jupyter notebook\category(1).csv"
    data = pd.read_csv(file,encoding="ISO-8859-1")
    category_id2id = {}
    category_id2id['root'] = 0
    for i in range(len(data)):
        category_id2id[data.loc[i, 'id']] = i + 1
    id2category_id = {value: key for key, value in category_id2id.items()}
    return category_id2id, id2category_id


def get_grid(city,num):
    grid_map={}
    file = "E:/LAND USE/data/"+city+"/"+city+"_"+str(num)+"cell_poi_10new#5.csv"
    category_id2id, id2category_id=get_category2id()
    data = pd.read_csv(file, encoding="ISO-8859-1")
    for i in tqdm(range(len(data))):
        category_id=data.loc[i,"category_id"]
        if category_id not in category_id2id:
            continue
        cell_id=data.loc[i,"cell_id"]
        x=int(data.loc[i,"x"])
        y =int( data.loc[i, "y"])
        if cell_id not in grid_map:
            grid_map[cell_id]=list(np.zeros((int(num/10),int(num/10),len(category_id2id))))
        grid_map[cell_id][x][y][category_id2id[category_id]]+=1
    return grid_map


if __name__ =="__main__":
    # cities=["Milano", "Amsterdam", "Barcelona", "Lisboa"]
    # lengths=[200, 300, 400, 500]
    cities=["Milano"]
    lengths=[200]
    min_count=5
    for city in cities:
        for length in lengths:
            grid_map=get_grid(city,length)
    print(grid_map)