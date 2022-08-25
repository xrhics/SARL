from math import radians, cos, sin, asin, sqrt,ceil,floor
import pandas as pd
import numpy as np
from tqdm import tqdm


def getDistance(lon1, lat1, lon2, lat2):  

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  
    return c * r * 1000

def get_mean_location(url,city_name):
    file=url+city_name+"/"+city_name+"_lonla.csv"
    data=pd.read_csv(file,encoding='utf-8')
    print("数据读取完毕")
    loc_map={}
    id_loc=[]
    id2info={}
    for i in tqdm(range(len(data))):
        if data.loc[i,"fid"] not in loc_map:
            loc_map[data.loc[i,"fid"]]=[[data.loc[i,"X"],data.loc[i,"Y"]]]  ##lon,la
            id2info[data.loc[i,"fid"]]=[data.loc[i,"area"],data.loc[i,"class_2018"]]
        else:
            loc_map[data.loc[i,"fid"]].append([data.loc[i,"X"],data.loc[i,"Y"]])
    print("数据载入完毕")
    for i in loc_map:
        array=np.array(loc_map[i])
        id_loc.append([i]+array.mean(axis=0).tolist()+id2info[i])   
    title=["id","lon","la","area","landuse"]
    info_data=pd.DataFrame(id_loc)
    info_data.to_csv("E:/LAND USE/data/"+city_name+"/"+city_name+"_landuse.csv",header=title,index=False)

def get_loca(url,city_name):
    file=url+city_name+"/"+city_name+"_lonla.csv"
    data=pd.read_csv(file,encoding='utf-8')
    print([data.loc[1,"X"],data.loc[1,"Y"]])

def pre_data():
    url = "E:/LAND USE/data/"
    city = ["Milano", "Amsterdam", "Barcelona", "Lisboa"]
    # city=["Amsterdam","Barcelona","Lisboa"]
    for city_name in city:
        get_mean_location(url,city_name)
        get_loca(url, city_name)


def get_ceilnum(num):
    return ceil(num * 100000) / 100000


def get_floornum(num):
    return floor(num * 100000) / 100000


def get_scope(url, city_name):
    file = url + city_name + "/" + city_name + "_landuse.csv"
    scope = []  # [[minlon,maxlon],[minla,maxla]]
    data = pd.read_csv(file, encoding='utf-8')
    scope.append([get_floornum(min(data["lon"])), get_ceilnum(max(data["lon"]))])
    scope.append([get_floornum(min(data["la"])), get_ceilnum(max(data["la"]))])
    return scope


def get_cell(url, city_name, scope, length):
    # file = url + city_name + "/" + city_name + "_landuse.csv"
    lon = 0.0012 * length / 100
    la = 0.0009 * length / 100
    if city_name == "Milano":
        lon = 0.0013 * length / 100
    if city_name == "Amsterdam":
        lon = 0.0015 * length / 100
    if city_name == "Barcelona":
        lon = 0.0012 * length / 100
    if city_name == "Lisboa":
        lon = 0.00115 * length / 100
    # la=0.0018  
    minlon = scope[0][0]
    maxlon = scope[0][1]
    minla = scope[1][0]
    maxla = scope[1][1]
    lon_num = ceil((maxlon - minlon) / lon)
    la_num = ceil((maxla - minla) / la)
    cell_list = []
    for i in tqdm(range(lon_num)):
        for j in range(la_num):
            cell_list.append([int(i), int(j), get_ceilnum(minlon + i * lon), get_floornum((minlon + (i + 1) * lon)),
                              get_ceilnum(minla + j * la), get_floornum(minla + (j + 1) * la)])
    title = ["x", "y", "minlon", "maxlon", "minla", "maxla"]
    cell_list_data = pd.DataFrame(cell_list, columns=title)
    print(111)
    cell_list_data.to_csv("E:/LAND USE/data/" + city_name + "/" + city_name + "_" + str(length) + "gridcell.csv",
                          index=False)
    return cell_list_data


def get_cell_map(data):
    cell_map = {}
    for i in range(len(data)):
        x = int(data.loc[i, 'x'])
        y = int(data.loc[i, 'y'])
        cell_map[x, y] = i
    return cell_map


def get_building_map(url, city_name):
    file = "E:/LAND USE/data/" + city_name + "/" + city_name + "_landuse.csv"
    building_map = {}
    data = pd.read_csv(file, encoding='utf-8')
    for i in range(len(data)):
        building_map[data.loc[i, "id"]] = [data.loc[i, "lon"], data.loc[i, "la"], data.loc[i, "area"],
                                           data.loc[i, "landuse"]]
    return building_map


def get_land_id(city_name, scope, building_lon, building_la, cell_map, length):
    lon = 0.0012 * length / 100
    la = 0.0009 * length / 100
    if city_name == "Milano":
        lon = 0.0013 * length / 100
    if city_name == "Amsterdam":
        lon = 0.0015 * length / 100
    if city_name == "Barcelona":
        lon = 0.0012 * length / 100
    if city_name == "Lisboa":
        lon = 0.00115 * length / 100
    minlon = scope[0][0]
    minla = scope[1][0]
    x = floor((building_lon - minlon) / lon)
    y = floor((building_la - minla) / la)
    return cell_map[x, y]


def get_cell_category(city_name, scope, cell_map, building_map, type2id, length):
    cell_land_map = {}
    for key in building_map:
        land_id = get_land_id(city_name, scope, building_map[key][0], building_map[key][1], cell_map, length)
        if land_id not in cell_land_map:
            cell_land_map[land_id] = [building_map[key][2:4]]
        else:
            cell_land_map[land_id].append(building_map[key][2:4])
    id2land_category = {}
    for i in tqdm(cell_land_map):
        land_area = {}
        for j in cell_land_map[i]:
            if type2id[j[1]] not in land_area:
                land_area[type2id[j[1]]] = j[0]
            else:
                land_area[type2id[j[1]]] += j[0]
        ranklist = sorted(land_area.items(), key=lambda x: x[1], reverse=True)
        for k in ranklist:
            land_id = k[0]
            area = k[1]
            break
        all_area = [sum(x) for x in zip(*ranklist)][1]
        if area / all_area >= 0.25:
            id2land_category[i] = land_id
    return id2land_category


def save_cell(data, city_name, id2land_category, length):
    cell_info = []
    for i in range(len(data)):
        cell_id = i
        if cell_id not in id2land_category:
            continue
        cell_value = data.loc[i].tolist()
        cell_value.append(id2land_category[cell_id])
        cell_info.append(cell_value)
    title = ["x", "y", "minlon", "maxlon", "minla", "maxla", "landuse"]
    cell_info_data = pd.DataFrame(cell_info, columns=title)
    cell_info_data.to_csv("E:/LAND USE/data/" + city_name + "/" + city_name + "_" + str(length) + "land2type.csv",
                          index=False)
    # real_map = {1: 1, 2: 2, 3: 1, 4: 3, 5: 2, 6:6,10: 4, 12: 5, 13: 5, 14: 2, 15: 5, 17: 5, 19: 4, 20: 4, 21: 4, 23: 4,25: 5}
    real_map = {1: 1, 2: 2, 3: 1, 4: 3, 5: 2, 10: 4, 12: 5, 13: 5, 14: 2, 15: 5, 17: 5, 19: 4, 20: 4, 21: 4, 23: 4,
                25: 5}
    for i in range(len(cell_info_data)):
        if cell_info_data.loc[i, "landuse"] not in real_map:
            cell_info_data = cell_info_data.drop(index=i)
            continue
        cell_info_data.loc[i, "landuse"] = real_map[cell_info_data.loc[i, "landuse"]]
    title = ["x", "y", "minlon", "maxlon", "minla", "maxla", "landuse"]
    cell_info_data.to_csv("E:/LAND USE/data/" + city_name + "/" + city_name + "_" + str(length) + "land2type$5.csv",
                          header=title, index=False)


def get_type2id():
    file = "E:/LAND USE/data/Lisboa/Lisboa_landuse.csv"
    data = pd.read_csv(file, encoding='utf-8')
    type2id = {}
    type_list = []
    n = 1
    for i in range(len(data)):
        if data.loc[i, "landuse"] not in type2id:
            type2id[data.loc[i, "landuse"]] = n
            type_list.append([n, data.loc[i, "landuse"]])
            n += 1
    title = ["id", "type"]
    type_data = pd.DataFrame(type_list)
    type_data.to_csv("E:/LAND USE/data/type_list.csv", header=title, index=False)
    return type2id

def save_cities_cell(city,length):
    url = "E:/LAND USE/data/"
    # city = ["Milano", "Amsterdam", "Barcelona", "Lisboa"]
    type2id = get_type2id()
    # length = [200, 300, 400, 500]
    for city_name in city:
        for i in length:
            print(city_name+" "+str(i))
            scope = get_scope(url, city_name)
            data = get_cell(url, city_name, scope, i)
            cell_map = get_cell_map(data)
            building_map = get_building_map(url, city_name)
            id2land_category = get_cell_category(city_name, scope, cell_map, building_map, type2id, i)
            save_cell(data, city_name, id2land_category, i)


if __name__ =="__main__":
    # pre_data()
    # length = [10, 200, 300, 400, 500]
    length = [10]
    city = ["Milano", "Amsterdam", "Barcelona", "Lisboa"]
    save_cities_cell(city,length)
