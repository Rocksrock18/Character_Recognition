from DataReader import DataReader
from NeuralNet_JIT import NeuralNet_JIT
from Population import Population
from Connection_JIT import Connection_JIT
from Predictor import Predictor
import random
import numba
import csv
import numpy as np
import pickle
from os import path

#file_name = "letter_num_dataset/emnist-balanced-train.csv"
#new_file_name = "letter_num_dataset/mini-subset.csv"
#with open(file_name) as csv_file:
#    data_list = list(csv.reader(csv_file))

#left = []
#for i in range(47):
#    left.append(3)

#with open(new_file_name, 'a', newline='') as csvfile:
#    csvwriter = csv.writer(csvfile)
#    for i in range(len(data_list)):
#        row = data_list[i]
#        if left[int(row[0])] > 0:
#            csvwriter.writerow(row)
#            left[int(row[0])] -= 1

#input("Done")

connection_type = Connection_JIT(0,0,0)
kv_ty = (numba.types.containers.UniTuple(numba.int64, 2), numba.typeof(connection_type))
master_connection_dict = numba.typed.Dict.empty(*kv_ty)
innov_list = numba.typed.List()
innov_list.append(1)

print("Begin test...")
pr = Predictor()
p = Population(784, 47, 1, master_connection_dict, innov_list, 0)
dr = DataReader()
mapping = dr.get_mapping()
images = dr.get_images(112800, 28, 28) # 112800 images in data set, each image 28x28 pixels
mp = p.networks[0]
for q in range(200):
    print("Adding connection " + str(q+1), end='\r')
    mp.add_connection()

#mp.randomize_all_bias()

print("\nStarting predictions...")
engines = [[mp, 0, 0]]

pr.make_predictions(engines, mapping, images, 1)
base = engines[0][2]
start_score = base

for i in range(1000):
    print("\nEpoch "+ str(i+1))
    pr.make_predictions(engines, mapping, images, 1)
    non_jit = p.construct_non_jit(mp)
    pickle.dump(non_jit, open("neural_net.txt", "wb"))

#end_score = engines[0][2]
#print("\nStart score = " + str(start_score) + ", End score = " + str(end_score))
#input("Done!")

##len(mp.connection_list)
#start = 5000 #next start = previous end = 5000
#end = 6000
#if path.exists("skip_list.txt"):
#    skip_list = pickle.load(open("skip_list.txt", "rb"))
#else:
#    skip_list = set()
##print(skip_list)
##print(len(skip_list))
#input("\nPress enter to start")
#for i in range(start, end):
#    if i not in skip_list:
#        rand = np.random.randn()
#        c = mp.connection_list[i]
#        c.weight += rand
#        print("\nEpoch "+ str(i+1))
#        pr.make_predictions(engines, mapping, images, 1)
#        if engines[0][2] < base:
#            c.weight -= rand
#        elif engines[0][2] == base:
#            c.weight -= rand
#            skip_list.add(i)
#        else:
#            base = engines[0][2]
##for q in range(5):
##    for i in range(len(mp.node_list)):
##        rand = np.random.randn()
##        n = mp.node_list[i]
##        if n.type != "Input":
##            n.bias += rand
##            print("\nEpoch "+ str(i+1))
##            pr.make_predictions(engines, mapping, images, 10)
##            if engines[0][2] <= base:
##                n.bias -= rand
##            else:
##                base = engines[0][2]
#end_score = base
#print("\nStart score = " + str(start_score) + ", End score = " + str(end_score))
#non_jit = p.construct_non_jit(mp)
#pickle.dump(non_jit, open("neural_net.txt", "wb"))
#pickle.dump(skip_list, open("skip_list.txt", "wb"))
#input("Finished tuning session")

for i in range(len(images)):
    image = images[i]
    #print("Testing image " + str(num+1), end='\r')
    inputs = pr.get_inputs(image[1], i)
    #print(inputs)
    result = mp.forward_prop(inputs)
    prediction = result.index(max(result))
    for i, r in enumerate(result):
        print("Amount for character " + chr(mapping[str(i)]) + ": " + str(r))
    print("\nPredicted " + str(chr(mapping[str(prediction)])) + ", correct character is " + chr(mapping[image[0]]))
    input()
input("\nDone")

print("\n\n===================================")
print("   Image Recognition on Characters")
print("===================================\n")

print("-----------------------------------\n       Simulation Parameters\n-----------------------------------\n")

pop_size = input(" - How many networks would you like in each generation? (Recommended: 300-500) ")
num_generations = input(" - How many generations would you like to simulate? (Recommended: 50-100) ")
print_best = input(" - Do you want to print the best network after each generation? (Y/N) ")
is_print = print_best == "Y" or print_best == "y"
num_hidden = 500

print("\n\n-----------------------------------\n        General Information\n-----------------------------------\n")

print("\t- A neural network is saved and updated for you automatically after each generation.\n\t- If you've run this simulation before, it'll be included in the starting population.")
input("\nPress [Enter] when you are ready to begin the simulation")

print("\n\n-----------------------------------\n         Simulation Start\n-----------------------------------\n")

print("Preparing the starting population. This will only take a moment...")

pop = Population(784, 47, int(pop_size), master_connection_dict, innov_list, num_hidden) # each image is 28x28 pixels, with 46 possible characters


print("Starting population complete!\t\t")

print("\n\n===================================\n    Simulating " + num_generations + " generation(s)\n===================================\n\n")

pop.simulate_generations(int(num_generations), is_print)

best = pop.networks[0]

best.show_net()

