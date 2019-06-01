import os

def clean_train(train_dir,valid_dir,model_names=[],pho_sizes=[],save_dirs=[],pre_trains=[],batch_sizes=[],denoise='',random_layer=0):
    num=len(model_names)
    for i in range(num):
        commond="python train.py --mode=\"train\" --train_dir=\"{}\" --valid_dir=\"{}\" --model_name=\"{}\" --pho_size={} --save_dir={} --pre_train={} --batch_size={} \
        --denoise={} --random_layer={}".format(train_dir,valid_dir,model_names[i],pho_sizes[i],save_dirs[i],pre_trains[i],batch_sizes[i],denoise,random_layer)
        print(commond)
        os.system(commond)
def HGD_train(train_dir,valid_dir,model_names=[],pho_sizes=[],save_dirs=[],pre_trains=[],batch_sizes=[],denoise='',random_layer=0):
    num=len(model_names)
    for i in range(num):
        commond="python train.py --mode=\"train_HGD\" --train_dir=\"{}\" --valid_dir=\"{}\" --model_name=\"{}\" --pho_size={} --save_dir={} --pre_train={} --batch_size={} \
        --denoise={} --random_layer={}".format(train_dir,valid_dir,model_names[i],pho_sizes[i],save_dirs[i],pre_trains[i],batch_sizes[i],denoise,random_layer)
        print(commond)
        os.system(commond)
def adversarial_train(train_dir,valid_dir,adversarial_train_dirs,adversarial_valid_dirs,model_names=[],pho_sizes=[],save_dirs=[],batch_sizes=[],adversarial_batch_sizes=[]):
    num=len(model_names)
    for i in range(num):
        commond="python train.py --mode=\"train\" --train_dir=\"{}\" --valid_dir=\"{}\" --adversarial_train_dirs=\"${}\" --adversarial_valid_dirs=\"{}\" --model_name=\"{}\" \
        --pho_size={} --save_dir={} --pre_train={} --batch_size={} --adversarial_batch_size={}".format(train_dir,valid_dir,adversarial_train_dirs,adversarial_valid_dirs,
            model_names[i],pho_sizes[i],save_dirs[i],pre_trains[i],batch_sizes[i],adversarial_batch_sizes[i])
        os.system(commond)

def generate_attack_images(input_dirs=[],model_names=[],pho_sizes=[],save_dirs=[],pre_trains=[],batch_sizes=[],adversarial_methods=[],eps=[],iterations=[]):
    num=len(model_names)
    print(num)
    for i in range(num):
        commond="python train.py --mode=\"generate\" --input_dir=\"{}\" --model_name=\"{}\" --pho_size={} --save_dir={} --pre_train=\"{}\" --batch_size={} \
        --adversarial_method=\"{}\" --eps={}  --iteration={}".format(input_dirs[i],model_names[i],pho_sizes[i],save_dirs[i],pre_trains[i],batch_sizes[i],
            adversarial_methods[i],eps[i],iterations[i])
        print("run {}".format(commond))
        os.system(commond)
def valid(input_dirs=[],model_names=[],pho_sizes=[],save_dirs=[],pre_trains=[],batch_sizes=[],denoise='',random_layer=0):
    num=len(model_names)
    for i in range(num):
        commond="python train.py --mode=\"valid\"  --input_dir=\"{}\" --model_name=\"{}\" --pho_size={} --save_dir={} --pre_train=\"{}\" --batch_size={}".format(
            input_dirs[i],model_names[i],pho_sizes[i],save_dirs[i],pre_trains[i],batch_sizes[i])
       	os.system(commond)
def test(test_dirs=[],model_names=[],pho_sizes=[],save_dirs=[],pre_trains=[],batch_sizes=[],denoise='',random_layer=0):
    num=len(model_names)
    #args.test_dir,args.adversarial_test_dir,args.pho_size
    for i in range(num):
        commond="python train.py --mode=\"test\"  --test_dir=\"{}\" --model_name=\"{}\" --pho_size={} --save_dir={} --pre_train=\"{}\" \
        --batch_size={} --denoise={} --random_layer={}".format(test_dirs[i],model_names[i],pho_sizes[i],save_dirs[i],pre_trains[i],batch_sizes[i],denoise,random_layer)
        print("run",commond)
        os.system(commond)
#models=[["vgg16",],["resnet101",],"densenet" "InceptionResNetV2" "InceptionV4" "Inception3"]
model_weight_size=[
["vgg16","vgg16_clean_299_model.dat",299],

["densenet","densenet_clean_166_model.dat",166],
["densenet","densenet_clean_299_model.dat",299],

["resnet101","resnet101_clean_192_model.dat",192],
["resnet101","resnet101_clean_299_model.dat",299],

["InceptionResNetV2","InceptionResNetV2_clean_299_model.dat",299],

["Inception3","Inception3_clean_299_model.dat",299],

["InceptionV4","Inceptionv4_clean_299_model.dat",299],

["resnet101","resnet101_clean_224_model.dat",224],
["resnet101","resnet101_adversarial_224_model.dat",224],
["Inception3","Inception3_clean_299_randomLayer_model.dat",299]

]

inputs=[
"/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test",
"/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train"
]

adversarial_methods=["fgsm","step-ll","i-fgsm"]
#clean_train(train_dir,valid_dir,model_names=[],pho_sizes=[],save_dirs=[],pre_trains=[],batch_sizes=[],denoise='',random_layer=0):
#clean_train(train_dir=inputs[1],valid_dir=inputs[0],model_names=['resnet152'],pho_sizes=[299],save_dirs=['./weights'],pre_trains=[''],batch_sizes=[16],random_layer=0)

"""HGD_299_model.dat             resnet101_adversial_model.dat          vgg16_clean_299_model.dat"""
#generate_attack_images()

#for i in [4,5,6,7]:
#test(test_dirs=[j],adversarial_test_dirs=[adversarial_dir],model_names=[model_weight_size[w][0]],pho_sizes=[model_weight_size[w][2]],save_dirs=["./output"],
#                        pre_trains=["./weights/"+model_weight_size[w][1]],batch_sizes=[8],denoise='./weights/HGD_299_model.dat')

for i in [2,4,5,6,7]:
    for j in inputs[:]:
        for k in adversarial_methods[:1]:
            for e in [0.03,0.06,0.12,0.24,0.36,0.48]:
                adversarial_dir=j+"_"+model_weight_size[i][1].replace('.','_')+"_"+k+"_"+str(e).replace('.','_')
                generate_attack_images(input_dirs=[j],model_names=[model_weight_size[i][0]],pho_sizes=[model_weight_size[i][2]],save_dirs=[adversarial_dir],
                    pre_trains=["./weights/"+model_weight_size[i][1]],batch_sizes=[4],adversarial_methods=[k],eps=[e],iterations=[10])
os.system("sudo python train_rectifi.py")
#HGD_train(inputs[1],inputs[0],model_names=[model_weight_size[4][0]],pho_sizes=[model_weight_size[4][2]],save_dirs=['./weights'],
#    pre_trains=[[model_weight_size[4][1]]],batch_sizes=[4],denoise='./weights/Inception3_clean_299_HGD_model.dat',random_layer=1)
#test(test_dirs=[],adversarial_test_dirs=[],model_names=[],pho_sizes=[],save_dirs=[],pre_trains=[],batch_sizes=[],denoise='',random_layer=0):
#test(test_dirs=[j],model_names=[model_weight_size[i][0]],pho_sizes=[model_weight_size[i][2]],save_dirs=["./output"],
#    pre_trains=["./weights/"+model_weight_size[i][1]],batch_sizes=[8],denoise='')
#test(test_dirs=[inputs[0]],model_names=[model_weight_size[10][0]],pho_sizes=[model_weight_size[10][2]],save_dirs=["./output"],
#    pre_trains=["./weights/"+model_weight_size[10][1]],batch_sizes=[4],denoise='./weights/Inception3_clean_299_HGD_model.dat',random_layer=1)


#for i in [8,9]:
#    for j in inputs[:1]:
#        test(test_dirs=[j],model_names=[model_weight_size[i][0]],pho_sizes=[model_weight_size[i][2]],save_dirs=["./output"],
#            pre_trains=["./weights/"+model_weight_size[i][1]],batch_sizes=[8],denoise='')

"""
for i in [2,4,5,6,7]:
    for j in inputs[:1]:
        for k in adversarial_methods[:2]:
            for e in [0.03,0.06,0.09,0.12]:
                adversarial_dir=j+"_"+model_weight_size[i][1].replace('.','_')+"_"+k+"_"+str(e).replace('.','_')
                #print(pre_train)
                for w in [1,3]:
                    test(test_dirs=[j],adversarial_test_dirs=[adversarial_dir],model_names=[model_weight_size[w][0]],pho_sizes=[model_weight_size[w][2]],save_dirs=["./output"],
                        pre_trains=["./weights/"+model_weight_size[w][1]],batch_sizes=[8])

"""
"""
for i in [2,4,5,6,7]:
    for j in inputs[:1]:
        for k in adversarial_methods[:2]:
            for e in [0.03,0.06,0.09,0.12]:
                adversarial_dir=j+"_"+model_weight_size[i][1].replace('.','_')+"_"+k+"_"+str(e).replace('.','_')
                #print(pre_train)
                for w in [3]:
                    test(test_dirs=[j],adversarial_test_dirs=[adversarial_dir],model_names=[model_weight_size[w][0]],pho_sizes=[299],save_dirs=["./output"],
                        pre_trains=["./weights/"+model_weight_size[w][1]],batch_sizes=[2],denoise='./weights/HGD_299_model.dat')
"""


