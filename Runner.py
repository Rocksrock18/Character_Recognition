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

print("Finished initial compilation!")

connection_type = Connection_JIT(0,0,0)
kv_ty = (numba.types.containers.UniTuple(numba.int64, 2), numba.typeof(connection_type))
master_connection_dict = numba.typed.Dict.empty(*kv_ty)
innov_list = numba.typed.List()
innov_list.append(1)

pr = Predictor()
p = Population(784, 47, 1, master_connection_dict, innov_list, 0)
dr = DataReader()

print("Beginning data retrieval...")
mapping = dr.get_mapping()
images = dr.get_test_images(112800, 28, 28) # 112800 max images in data set, each image 28x28 pixels
mp = p.networks[0]

print("Finished! Now, testing the network on the dataset...")

#for i in range(3000):
#    mp.add_connection()

pr.set_affinity(mp, images)

#non_jit = p.construct_non_jit(mp)
#pickle.dump(non_jit, open("neural_net.txt", "wb"))
#input("Done")

engines = [[mp, 0, 0]]

for i in range(100):
    print("\nEpoch "+ str(i+1))
    pr.make_predictions(engines, mapping, images, 1)
    non_jit = p.construct_non_jit(mp)
    pickle.dump(non_jit, open("neural_net.txt", "wb"))

print("Finished")
