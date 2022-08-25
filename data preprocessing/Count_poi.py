from math import radians, cos, sin, asin, sqrt,ceil,floor
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pylab import *

def get_cell_map(city,length):
    grid_map={}
    id2type={}
    file="E:/LAND USE/data/"+city+"/"+city+"_"+str(length)+"land2type$5.csv"
    if city == "NYC":
        file = "E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "land2type(1).csv"
    data=pd.read_csv(file,encoding="ISO-8859-1")
    for i in range(len(data)):
        grid_map[data.loc[i,"x"],data.loc[i,"y"]]=i
        id2type[i]=data.loc[i,"landuse"]
    return grid_map,id2type

def get_poi(grid_map,id2type,data,city,length):
    info=[]
    for i in tqdm(range(len(data))):
        la=data.loc[i,"lat"]
        lon=data.loc[i,"lng"]
        if city=="NYC":
            y=floor((la-40.4961)/(0.0009*length/100))
            x=floor((lon-(-74.2557))/(0.0012*length/100))
        if city=="Milano":
            y=floor((la-45.1147)/(0.0009*length/100))
            x=floor((lon-8.5938)/(0.0013*length/100))
        if city=="Amsterdam":
            y=floor((la-52.1073)/(0.0009*length/100))
            x=floor((lon-4.506)/(0.0015*length/100))
        if city=="Barcelona":
            y=floor((la-41.1866)/(0.0009*length/100))
            x=floor((lon-1.5592)/(0.0012*length/100))
        if city=="Lisboa":
            y=floor((la-38.4156)/(0.0009*length/100))
            x=floor((lon-(-9.4983))/(0.00115*length/100))
        if (x,y) in grid_map:
            info.append(data.loc[i].tolist()+[grid_map[x,y],id2type[grid_map[x,y]]])
    print(len(info))
    title = [ "id","name", "lng", "lat", "category_id", "category_name","cell_id","cell_type"]
    data=pd.DataFrame.from_dict(info)
    data.to_csv("E:/LAND USE/data/"+city+"/" + city+ "_"+str(length)+"cell_poi.csv", header=title,index=False,encoding="utf-8")
    return info

def draw_CDF(count_venue_array,city):              ###画CDF图
    sample = np.array(count_venue_array)
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(0, 60)
    y = ecdf(x)
    figure(figsize=(10,6), dpi=108)
    subplot(1,1,1)
    plt.plot(x, y, linewidth = '3',label='first')
    # x = np.linspace(min(sample), max(sample))
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 28,
    }
    linewidth=2
    #zuobiaozhou daxiao
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth)
    ax.tick_params(length=6)
    x_major_locator=MultipleLocator(15)
    ymajorLocator   = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(ymajorLocator)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 44,
    }
    plt.xlabel('poi number',font2)
    plt.ylabel('radio less than poi num',font2)
    plt.title(city+'poi CDF',font2)
    savefig("E:/LAND USE/data/picture/"+city+"/"+city+"_poi_num_in_cell.pdf",dpi=108,bbox_inches = 'tight')
    plt.show()

def get_cell_poi_num(url):
    cell_poi_num={}
    data=pd.read_csv(url,encoding="ISO-8859-1")
    for i in range(len(data)):
        if data.loc[i,"cell_id"] not in cell_poi_num:
            cell_poi_num[data.loc[i,"cell_id"]]=1
        else:
            cell_poi_num[data.loc[i,"cell_id"]]+=1
    return cell_poi_num



def get_cell_poi(cities,length):
    # cities=["NYC","Milano","Amsterdam","Barcelona","Lisboa"]
    #cities = ["Milano", "Amsterdam", "Barcelona", "Lisboa"]
    # length = [200, 300, 400, 500]
    for city in cities:
        print(city)
        for i in length:
            url = "E:/LAND USE/data/" + city + "/" + city + "_poi.csv"
            data = pd.read_csv(url, encoding="ISO-8859-1")
            grid_map, id2type = get_cell_map(city, i)
            info = get_poi(grid_map,id2type, data, city, i)

def get_cell_info(cities,lengths):
    # cities=["NYC","Milano","Amsterdam","Barcelona","Lisboa"]
    # cities = ["Milano", "Amsterdam", "Barcelona", "Lisboa"]
    # lengths = [200, 300, 400, 500]
    for city in cities:
        print(city)
        for length in lengths:
            url = "E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "cell_poi.csv"
            cell_poi_num = get_cell_poi_num(url)
            # print(max(cell_poi_num.values()))
            poi_num_array = [cell_poi_num[key] for key in cell_poi_num]
            # draw_CDF(poi_num_array,city)
            file = "E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "land2type$5.csv"
            if city=="NYC":
                file = "E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "land2type(1).csv"
            data = pd.read_csv(file, encoding="ISO-8859-1")
            cell_list = []
            for i in range(len(data)):
                if i in cell_poi_num:
                    cell_list.append([i] + data.loc[i].tolist() + [cell_poi_num[i]])
            print(len(cell_list), sum(list(cell_poi_num.values())))
            title = ["id", "x", "y", "minlon", "maxlon", "minla", "maxla", "landuse", "poinum"]
            data = pd.DataFrame.from_dict(cell_list)
            data.to_csv("E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "cell_info.csv", header=title,
                        index=False, encoding="utf-8")

def category2parent_name(Path):
    data = pd.read_csv(Path)
    id2name={data.loc[i,"id"]:data.loc[i,"name"] for i in range(len(data))}
    category2parent={}
    for i in range(len(data)):
        name=data.loc[i,"name"]
        parent=data.loc[i,"parent"]
        if len(parent.split("$"))<=1:
            parent=name
        else:
            parent=id2name[parent.split("$")[-2]]
        category2parent[name]=parent
    return category2parent

def get_poi_info(cities,lengths):
    Path = "E:\jupyer notebook\deepmove\data\category.csv"
    category2parent =category2parent_name(Path)
    # cities=["NYC","Milano","Amsterdam","Barcelona","Lisboa"]
    for city in cities:
        for length in lengths:
            url = "E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "cell_poi.csv"
            data = pd.read_csv(url, encoding="ISO-8859-1")
            category_num = {}
            a = []
            for i in range(len(data)):
                if data.loc[i, "category_name"] not in category2parent:
                    # print(data.loc[i,"category_name"] )
                    if data.loc[i, "category_name"] not in a:
                        a.append(data.loc[i, "category_name"])
                    continue
                if category2parent[data.loc[i, "category_name"]] not in category_num:
                    category_num[category2parent[data.loc[i, "category_name"]]] = 1
                else:
                    category_num[category2parent[data.loc[i, "category_name"]]] += 1
            print(category_num)

if __name__ =="__main__":
    # length = [200, 300, 400, 500]
    length = [10]
    # cities = ["NYC", "Milano", "Amsterdam", "Barcelona", "Lisboa"]
    cities = [ "Milano", "Amsterdam", "Barcelona", "Lisboa"]
    get_cell_poi(cities,length)
    # get_cell_info(cities,length)
    get_poi_info(cities,length)

