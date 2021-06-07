import argparse
import sys
import os 

from os import listdir
parser = argparse.ArgumentParser()

parser.add_argument("--input_folder", default=None, type=str, required=True, help="path to the input_folder")
parser.add_argument("--output_folder", default=None, type=str, required=True,help="path to the out_folder")

parser.add_argument("--l_code_actual", default=None, type=str, required=True,help="language code")
parser.add_argument("--l_code_in_raw_data", default=None, type=str, required=True,help="language code")


parser.add_argument("--train_files_taken", default=None, type=str, required=True, help="path to the input_folder")
parser.add_argument("--valid_files_taken", default=None, type=str, required=True, help="path to the input_folder")
parser.add_argument("--test_files_taken", default=None, type=str, required=True, help="path to the input_folder")

args = parser.parse_args()

# files=os.listdir(args.input_folder)
try:
    os.mkdir(args.output_folder)
except:
    print('error while creating outdir',args.output_folder)

out_dir=args.output_folder+'/'+args.l_code_actual

try:
    os.mkdir(out_dir)
except:
    print('error while creating outdir',out_dir)

pos_tags_list=[t.split('\t')[0] for t in open('penn_to_bis_mapping.txt','r').readlines()[:-1]]


print(pos_tags_list)

dataset_type='train'
input_files=open(args.train_files_taken,'r').readlines()
print(input_files)
outfile=open(args.output_folder+'/'+args.l_code_actual+'/'+dataset_type+'-'+args.l_code_actual+'.tsv','w')
outfile.write('label\ttext\n')

for file in input_files:

	print("File ",file)
	f=args.input_folder+'/'+args.l_code_in_raw_data+'_'+file[:-1]+'.txt'
	# print(listdir(args.input_folder))
	
	f=open(f,'r')
	f.readline()
	a=f.readlines()
	print(len(a))
	file_class=file.split('_')[0]
	print(file,file_class)
	count1=0
	count2=0
	for sent in a:
		ws=[]
		ts=[] 

		words=sent.split('\n')[0].split('\t')[-1].split(']]') 
		for wg in words[:-1]: 
			for w in wg.split('[[')[-1].split(' '):

				word=w.split('\n')[0].split("\\")[0].split('\t')[-1]
				tags=w.split('\n')[0].split("\\")[-1].split('_') 
				t1=tags[0 % len(tags)] 
				t2=tags[1 %len(tags)]
				ws.append(word)
				ts.append(t1)

				if(t1 not in pos_tags_list):
					print(t1)
					ws=[]
					ts=[]
					break

			if(ws==[]):
				break	
		if(ws==[]):
			continue
		else:
			outfile.write(file_class+'\t'+' '.join(ws)+'\n')


	print(count1,count2)                                    
	f.close()
outfile.close()

dataset_type='valid'
input_files=open(args.valid_files_taken,'r').readlines()
print(input_files)
outfile=open(args.output_folder+'/'+args.l_code_actual+'/'+dataset_type+'-'+args.l_code_actual+'.tsv','w')
outfile.write('label\ttext\n')

for file in input_files:


	f=args.input_folder+'/'+args.l_code_in_raw_data+'_'+file[:-1]+'.txt'
	# print(listdir(args.input_folder))
	
	f=open(f,'r')
	f.readline()
	a=f.readlines()

	file_class=file.split('_')[0]
	print(file,file_class)
	count1=0
	count2=0
	for sent in a:
		ws=[]
		ts=[] 

		words=sent.split('\n')[0].split('\t')[-1].split(']]') 
		for wg in words[:-1]: 
			for w in wg.split('[[')[-1].split(' '):

				word=w.split('\n')[0].split("\\")[0].split('\t')[-1]
				tags=w.split('\n')[0].split("\\")[-1].split('_') 
				t1=tags[0 % len(tags)] 
				t2=tags[1 %len(tags)]
				ws.append(word)
				ts.append(t1)

				if(t1 not in pos_tags_list):
					ws=[]
					ts=[]
					break

			if(ws==[]):
				break	
		if(ws==[]):
			continue
		else:
			outfile.write(file_class+'\t'+' '.join(ws)+'\n')


	print(count1,count2)                                    
	f.close()
outfile.close()



dataset_type='test'
input_files=open(args.test_files_taken,'r').readlines()
outfile=open(args.output_folder+'/'+args.l_code_actual+'/'+dataset_type+'-'+args.l_code_actual+'.tsv','w')
outfile.write('label\ttext\n')

for file in input_files:


	f=args.input_folder+'/'+args.l_code_in_raw_data+'_'+file[:-1]+'.txt'
	# print(listdir(args.input_folder))
	f=open(f,'r')
	f.readline()
	a=f.readlines()
	print(args.output_folder+'/'+dataset_type+'-'+args.l_code_actual+'.tsv')
	file_class=file.split('_')[0]
	for sent in a:
		ws=[]
		ts=[] 

		words=sent.split('\n')[0].split('\t')[-1].split(']]') 
		for wg in words[:-1]: 
			for w in wg.split('[[')[-1].split(' '):

				word=w.split('\n')[0].split("\\")[0].split('\t')[-1]
				tags=w.split('\n')[0].split("\\")[-1].split('_') 
				t1=tags[0 % len(tags)] 
				t2=tags[1 %len(tags)]
				ws.append(word)
				ts.append(t1)

				if(t1 not in pos_tags_list):
					ws=[]
					ts=[]
					break

			if(ws==[]):
				break	
		if(ws==[]):
			continue
		else:
			outfile.write(file_class+'\t'+' '.join(ws)+'\n')

			# for i in range(len(ws)):
			# 	word=ws[i]
			# 	t1=ts[i]
			# 	outfile.write(word+'\t'+t1+'\n')



		# outfile.write('\n')                                    
	f.close()
outfile.close()	