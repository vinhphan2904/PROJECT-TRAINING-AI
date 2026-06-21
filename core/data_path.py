from lib import *

def data_list(phase = 'train'):
    dir_data = './data/dataset'
    path = os.path.join(dir_data,phase,'**','*.*')
    path_list = []
    for list in glob.glob(path,recursive=True):
        path_list.append(list)
    return path_list