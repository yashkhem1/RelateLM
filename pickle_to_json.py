import os
import pickle
import json
from tqdm import tqdm

# f = open('Hindi_Marathi.pkl','rb')
# w = open('Hindi_Marathi.json','a')

# a = pickle.load(f)
# _=w.write('{\n')
# _=w.write('\t"data":[\n')
# for i in tqdm(range(len(a['token_ids_f1']))):
#     if i ==0:
#         _=w.write('\t\t{"token_ids_1":'+str(a['token_ids_f1'][i])+',"token_ids_2":'+str(a['token_ids_f2'][i])+',"token_maps_1":'+str(a['token_maps_f1'][i])+',"token_maps_2":'+str(a['token_maps_f2'][i])+'}')
#     else:
#         _=w.write(',\n\t\t{"token_ids_1":'+str(a['token_ids_f1'][i])+',"token_ids_2":'+str(a['token_ids_f2'][i])+',"token_maps_1":'+str(a['token_maps_f1'][i])+',"token_maps_2":'+str(a['token_maps_f2'][i])+'}')

# _=w.write('\n\t]\n')
# _=w.write('}')
# f.close()
# w.close()

f = open('Marathi_Hindi.json','r')
a = json.load(f)
mh = a['data']
for i,datum in enumerate(mh):
    # if i < 10:
    #     print(datum['token_maps_2'])
    if len(datum['token_ids_2']) != 128:
        print("A")
        # print(datum['token_maps_2'])
        # print(len(datum['token_maps_2']))
        # print("B")
        # print(datum['token_ids_2'])
        # print(len(datum['token_ids_2']))
        # print("C")
        # print(datum['token_ids_1'])
        # print(len(datum['token_ids_1']))
