import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize


def get_category2id():
    file = "E:\jupyer notebook\category.csv"
    data = pd.read_csv(file,encoding="ISO-8859-1")
    category_id2id = {}
    category_id2id['root'] = 0
    for i in range(len(data)):
        category_id2id[data.loc[i, 'id']] = i + 1
    id2category_id = {value: key for key, value in category_id2id.items()}
    return category_id2id, id2category_id

def get_category2parent():
    file = "E:\jupyer notebook\category.csv"
    data = pd.read_csv(file,encoding="ISO-8859-1")
    category2parent = {}
    for i in range(len(data)):
        parent=data.loc[i, 'parent'].split("$")
        if len(parent) < 2:
            category2parent[data.loc[i, 'id']]=data.loc[i, 'id']
        else:
            category2parent[data.loc[i, 'id']] = parent[-2]
    return category2parent

def get_grid(city,num):
    grid_map={}
    file = "E:/LAND USE/data/"+city+"/"+city+"_"+str(num)+"cell_poi_10new#5.csv"
    category_id2id, id2category_id=get_category2id()
    category2parent=get_category2parent()
    data = pd.read_csv(file, encoding="ISO-8859-1")
    for i in tqdm(range(len(data))):
        category_id=data.loc[i,"category_id"]
        if category_id not in category_id2id:
            continue
        cell_id=data.loc[i,"cell_id"]
        x=int(data.loc[i,"x"])
        y =int( data.loc[i, "y"])
        if cell_id not in grid_map:
            grid_map[cell_id]=np.zeros((int(num/10),int(num/10),10),dtype='float32')
        grid_map[cell_id][x][y][category_id2id[category2parent[category_id]]-1]+=1
    return grid_map

class cell_data:
    def __init__(self,city,length,min_count):
        self.city=city
        self.length=length
        self.min_count=min_count
        self.file1= "E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "cell_point_tree#" + str(min_count) + "_new.csv"
        # self.file1="E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "cell_poi_tf-idf#" + str(min_count) + ".csv"
        self.file2="E:/LAND USE/data/" + city + "/" + city + "_" + str(length) + "cell_info.csv"
        self.cell_id2id={}
        self.id2cell_id={}
        self.cell_id2type={}
        self.num=0
        if city=="NYC":
            self.class_num=6
        else:
            self.class_num =5
        self.data,self.cell_id_list=None,None

    def get_data(self):
        grid_map=get_grid(self.city,self.length)
        cell_id2id={}
        cell_data=[]
        data = pd.read_csv(self.file2)
        self.cell_id2type = {data.loc[i, 'id']: data.loc[i, "landuse"] for i in range(len(data))}
        data = pd.read_csv(self.file1)
        x = data.drop(['0'], axis=1)
        self.num = len(x.loc[0])
        x=np.array(x).astype(np.float32).tolist()
        for i in range(len(data)):
            cell_id=data.loc[i,'0']
            cell_id2id[cell_id]=i
            cell_data.append([list(x[i]),list(grid_map[cell_id]),int(self.cell_id2type[cell_id]-1),self.onehot(int(self.cell_id2type[cell_id]))])  #这里把所有的类型编码-1
        self.cell_id2id=cell_id2id
        self.id2cell_id={cell_id2id[key]:key for key in cell_id2id}
        return cell_data,list(data['0'])

    def onehot(self,type):
        vec = np.zeros(self.class_num, dtype=np.float32)
        vec[type - 1] = 1
        return list(vec)




if __name__ == '__main__':
       city = "Milano"
       length = 200
       min_count = 5
       cell=cell_data(city,length,min_count)
       print(cell.data)
