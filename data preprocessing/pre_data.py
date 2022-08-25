
import pandas as pd
from tqdm import tqdm

def get_maps(file):
    data = pd.read_csv(file,encoding="ISO-8859-1")
    id2poi={}
    cell_category={}
    category_id2id,id2category_id,id2parent_id=get_category2id()
    for i in tqdm(range(len(data))):
        category_id=data.loc[i,"category_id"]
        if category_id in category_id2id:
            id2poi[i]=data.loc[i].tolist()
            cell=data.loc[i,"cell_id"]
            if cell not in cell_category:
                cell_category[cell]=[0 for i in range(len(category_id2id))]
            cell_category[cell][category_id2id[category_id]]+=1
    return id2poi,cell_category,category_id2id,id2category_id,id2parent_id

def get_cell_type(file):
    data=pd.read_csv(file)
    cell_type={}
    for i in range(len(data)):
        cell_type[data.loc[i,"id"]]=data.loc[i,"landuse"]
    return cell_type

def get_category2id():
    file = "E:\jupyer notebook\category.csv"
    data = pd.read_csv(file,encoding="ISO-8859-1")
    category_id2id = {}
    id2parent={}
    category_id2id['root'] = 0
    for i in range(len(data)):
        category_id2id[data.loc[i, 'id']] = i + 1
        parents=data.loc[i,"parent"].split("$")
        id2parent[i+1]=[category_id2id[parent] for parent in parents]
    id2category_id = {value: key for key, value in category_id2id.items()}
    return category_id2id, id2category_id,id2parent



def save_tree(city,num,min_count):
    file = "E:/LAND USE/data/"+city+"/"+city+"_"+str(num)+"cell_poi_10new.csv"
    id2poi, cell_category, category_id2id, id2category_id,id2parent_id = get_maps(file)
    for i in list(cell_category.keys()):
        if sum(cell_category[i])<min_count:
            del cell_category[i]
    tree_list=[]
    for i in tqdm(cell_category):
        tree_line=[0 for j in range(len(cell_category[i]))]
        for j in range(len(cell_category[i])):
            if cell_category[i][j]==0:
                continue
            tree_line[j]+=cell_category[i][j]
            for k in id2parent_id[j]:
                tree_line[k]+=cell_category[i][j]
        tree_list.append([i] + tree_line)
    tree_list_df=pd.DataFrame(tree_list)
    title=[i for i in range(len(tree_line)+1)]
    tree_list_df.to_csv("E:/LAND USE/data/" + city + "/" + city + "_" + str(num) + "cell_point_tree#"+str(min_count)+"_new.csv", index=False,header=title)


if __name__ =="__main__":
    # file="E:\LAND USE\data\\NYC\\NYC_200cell_poi.csv"
    # id2poi,cell_category,category_id2id,id2category_id=get_maps(file)
    # print(cell_category[94])
    # my_tree=zone_tree(category_id2id)
    # my_tree.load(cell_category[94])
    # print(my_tree.get_list())
    cities=["Milano", "Amsterdam", "Barcelona", "Lisboa"]
    lengths=[200, 300, 400, 500]
    min_count=5
    for city in cities:
        for length in lengths:
            print(city+"  "+str(length))
            save_tree(city, length,min_count)
