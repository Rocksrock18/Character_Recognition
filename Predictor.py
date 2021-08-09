import PIL
from PIL import Image
import random
import math
import numpy as np
import time

class Predictor():
    def __init__(self):
        self.image_dict = {}

    def get_inputs_from_formatted(self, pixels, key):
        if key in self.image_dict:
            return self.image_dict[key]
        inputs = np.zeros(784)
        for i, column in enumerate(pixels):
            for k, pixel in enumerate(column):
                inputs[i*28 + k] = pixel/255
        self.image_dict[key] = inputs
        return inputs

    def get_inputs(self, pixels, key):
        return pixels

    def set_affinity(self, net, images):
        output_dict = {}
        for node in net.node_list:
            print("Output_dict for node id: " + str(node.id) + "\t", end='\r')
            output_dict[node.id] = self.get_output_connections(node, net)
        for key in net.node_order_dict:
            n = net.find_node(key)
            net_affinity = 0
            for c in n.receptions:
                print("Calc affinity for connection: " + str(c.innov) + "\t", end='\r')
                res = self.calc_affinity(c, net, images, output_dict[c.output_id])
                c.affinity = res
                net_affinity += res
            n.affinity = net_affinity/max(1, len(n.receptions))

    def calc_affinity(self, c, net, images, output_list):
        affinity = 0
        last_input = 784
        input_node = net.find_node(c.input_id)
        for i, image in enumerate(images):
            correct = int(image[0])
            inputs = self.get_inputs(image[1], i)
            for o in output_list:
                if (o-last_input-1) == correct:
                    if input_node.type == "Input":
                        affinity += inputs[c.input_id - 1] * 46
                    else:
                        affinity += input_node.affinity * 46
                else:
                    if input_node.type == "Input":
                        affinity -= inputs[c.input_id - 1]
                    else:
                        affinity -= input_node.affinity
        return affinity/len(images)


    def get_output_connections(self, node, net):
        output_list = np.zeros(47)
        out_set = set()
        next_index = 0
        if node.type == "Output":
            output_list[0] = node.id
            return output_list
        nodes_to_check = []
        nodes_to_check.append(node)
        while len(nodes_to_check) > 0:
            n = nodes_to_check[0]
            for c in n.connections:
                output_n = net.find_node(c.output_id)
                if output_n.type == "Output":
                    if output_n.id not in out_set:
                        output_list[next_index] = output_n.id
                        next_index += 1
                        out_set.add(output_n.id)
                        if len(out_set) == 47:
                            return output_list
                else:
                    nodes_to_check.append(output_n)
            nodes_to_check.pop(0)
        return output_list

    
    def make_predictions(self, nets, mapping, images, scalar):
        temp = []
        for q in range(47):
            temp.append(0)
        for index, net in enumerate(nets):
            total = 0
            num_correct = 0
            score = 0
            total_score = 0
            correction = np.zeros(47)
            for i in range(len(images)):
                #ind = np.random.randint(0, 112800)
                image = images[i]
                #print("Testing image " + str(num+1), end='\r')
                inputs = self.get_inputs(image[1], i)
                #print(inputs)
                result = net[0].forward_prop(inputs)
                #result = temp
                prediction = result.index(max(result))
                #for i, r in enumerate(result):
                #    print("Amount for character " + chr(mapping[str(i)]) + ": " + str(r))
                #print("\nPredicted " + str(chr(mapping[str(prediction)])) + ", correct character is " + chr(mapping[image[0]]))
                #img = Image.fromarray(np.array(image[1], dtype=np.uint8))
                #img.show()
                if prediction == int(image[0]):
                    num_correct += 1
                total += 1
                for ind, r in enumerate(result):
                    if ind == int(image[0]):
                        score += 46*r
                        total_score += 46
                        correction[ind] = 46*(1-r)
                    else:
                        score += 1 - r
                        total_score += 1
                        correction[ind] = -r
                #input("Check")
                #for k, c in enumerate(correction):
                #    net[0].find_node(785 + k).correction = correction[k]
                #net[0].back_prop()
                #print("Currently has a score of " + str(score/total_score) + " and accuracy of " + str(round((num_correct/total)*100,3)) + "%\t\t", end='\r')
            #print(correction)
            print("Finished testing net " + str(index+1) + " with score of " + str(score/total_score) + " and accuracy of " + str(round((num_correct/total)*100,3)) + "%\t\t\t", end='\r')
            #input()
            base = num_correct/total
            base_score = score/total_score
            net[1] = base
            net[2] = base_score**scalar
