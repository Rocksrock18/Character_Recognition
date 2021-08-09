from DataReader import DataReader
from Population import Population
from Connection_JIT import Connection_JIT
from Predictor import Predictor
import numba
import numpy as np
import time

def reg_matmul(m1, m2):
    return np.matmul(m1, m2)

a = 1
b = 1
c = 1
d = 1

matrix1 = np.zeros((1,2))
matrix2 = np.zeros((2,1))

matrix1[0,0] = a
matrix1[0,1] = b
matrix2[0,0] = c
matrix2[1,0] = d


def get_dummy_list():
    new_list = numba.typed.List()
    new_list.append(np.zeros(784))
    return new_list


def get_answers(images):
    answers = np.zeros(len(images))
    for i, image in enumerate(images):
        answers[i] = image[0]
    return answers

def get_pixels(images):
    answers = get_dummy_list()
    answers.pop(0)
    for i, image in enumerate(images):
        answers.append(image[1])
    return answers

def affinity_optimized(net, images, pr):
    output_dict = {}
    images_answers = get_answers(images)
    images_pixels = get_pixels(images)
    for node in net.node_list:
        #print("Output_dict for node id: " + str(node.id) + "\t", end='\r')
        output_dict[node.id] = pr.get_output_connections(node, net)
    for key in net.node_order_dict:
        n = net.find_node(key)
        net_affinity = 0
        for c in n.receptions:
            #print("Calc affinity for connection: " + str(c.innov) + "\t", end='\r')
            res = calc_affinity_optimized(c, net, images, images_answers, images_pixels, output_dict[c.output_id], pr)
            c.affinity = res
            net_affinity += res
        n.affinity = net_affinity/max(1, len(n.receptions))

def calc_affinity_optimized(c, net, images, images_answers, images_pixels, output_list, pr):
    input_node = net.find_node(c.input_id)
    if input_node.type == "Input":
        #return calc_affinity_input(c, net, images, output_list, pr, input_node)
        return net.calc_affinity_input(c, net, images_answers, images_pixels, output_list, input_node) #, net, images_answers, images_pixels, output_list, input_node
    return calc_affinity_non_input(c, net, images, output_list, pr, input_node)

def calc_affinity_input(c, net, images, output_list, pr, input_node):
    affinity = 0
    last_input = 784
    for i, image in enumerate(images):
        correct = int(image[0])
        inputs = image[1]
        for o in output_list:
            #affinity = affinity + inputs[c.input_id - 1] * 46 if (o-last_input-1) == correct else affinity - inputs[c.input_id-1]
            if (o-last_input-1) == correct:
                affinity += inputs[c.input_id - 1] * 46
            else:
                affinity -= inputs[c.input_id - 1]
    return affinity/len(images)

def calc_affinity_non_input(c, net, images, output_list, pr, input_node):
    affinity = 0
    last_input = 784
    for i, image in enumerate(images):
        correct = int(image[0])
        inputs = image[1]
        for o in output_list:
            if (o-last_input-1) == correct:
                affinity += input_node.affinity * 46
            else:
                affinity -= input_node.affinity
    return affinity/len(images)


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



res1 = reg_matmul(matrix1, matrix2)
print(res1)
res2 = mp.test_matmul(matrix1, matrix2)
print(res2)

input("Check")

engines = [[mp, 0, 0]]
#pr.make_predictions(engines, mapping, images, 1)
start_time = time.time()
for r in range(1000):
    #pr.make_predictions(engines, mapping, images, 1)
    mp.test_matmul(matrix1, matrix2)
end_time = time.time()
print("Took average of " + str((end_time - start_time) / 10) + " seconds.\t\t")
