from NeuralNet_JIT import NeuralNet_JIT
from NeuralNet import NeuralNet
from Connection_JIT import Connection_JIT
from Node_JIT import Node_JIT
from Predictor import Predictor
from DataReader import DataReader
from DataWriter import DataWriter
import math
import numpy as np
import numba
import pickle
import time
import copy
from os import path


class Population():
    def __init__(self, num_inputs, num_outputs, pop_size, mcd, innov_list, num_hidden_nodes):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.pop_size = pop_size
        self.struct_scale = 0.5
        self.weight_scale = 3
        self.diff_threshold = 2.5
        should_copy, master_parent = self.get_master_parent(num_inputs, num_outputs, num_hidden_nodes, mcd, innov_list)
        self.networks, self.species = self.populate(master_parent, should_copy, mcd, innov_list, num_hidden_nodes)

    def populate(self, master_parent, should_copy, mcd, innov_list, num_hidden_nodes):
        networks = []
        if should_copy:
            jit = self.construct_jit(master_parent, mcd, innov_list)
            networks.append(jit)
        else:
            networks.append(master_parent)
        for i in range(self.pop_size-1):
            print("Creating member " + str(i + 2) + "/" + str(self.pop_size) + "\t", end='\r')
            if should_copy:
                net = self.construct_jit_copy(networks[0], mcd, innov_list)
            else:
                net = NeuralNet_JIT(self.num_inputs, self.num_outputs, mcd, innov_list)
                net.reset_neural_net()
            for j in range(num_hidden_nodes):
                net.add_node()
            net.mutate(1)
            networks.append(net)
        species = self.assign_pop_to_unique_species(networks)
        return networks, species

    def assign_pop_to_unique_species(self, pop):
        species = []
        for net in pop:
            species.append([net])
        return species

    def get_master_parent(self, num_inputs, num_outputs, num_hidden_nodes, mcd, innov_list):
        if path.exists("neural_net.txt"):
            old_net = pickle.load(open("neural_net.txt", "rb"))
            #jit = self.construct_jit(old_net, mcd, innov_list)
            #jit.master_reset_innov()
            old_net.master_reset_innov()
            return True, old_net
        net = NeuralNet_JIT(num_inputs, num_outputs, mcd, innov_list)
        net.reset_neural_net()
        #net.generate_hidden_layers(self.test_layers)
        for j in range(num_hidden_nodes):
            net.add_node()
        return False, net

        
        
    def sum_fitness(self, s):
        sum = 0
        for net in s:
            sum += net.fitness
        return sum

    def avg_fitness(self, s):
        return self.sum_fitness(s) / len(s)


    def find_best_member(self, s):
        best_fitness = s[0].fitness
        best_member = s[0]
        for net in s:
            if net.fitness > best_fitness:
                best_member = net
                best_fitness = net.fitness
        return best_member



    def choose_parent(self, s):
        threshold = np.random.random() * self.sum_fitness(s)
        sum = 0
        for net in s:
            sum += net.fitness
            if sum >= threshold:
                return net
        return s[len(s)-1]



    def assign_pop_to_species(self, pop, species):
        for i, net in enumerate(pop):
            print("Assigning member " + str(i + 1) + "/" + str(self.pop_size) + " to a species\t", end='\r')
            species_found = False
            for s in species:
                if not species_found:
                    if(len(s) > 0):
                        rep = s[np.random.randint(0, len(s))]
                        diff = NeuralNet_JIT.calc_difference(net, rep, self.weight_scale, self.struct_scale)
                        if(diff < self.diff_threshold):
                            s.append(net)
                            species_found = True
                else:
                    break
            if not species_found:
                species.append([net])
        return species


    def determine_connection(self, i, p1, p2):
        c1 = p1.find_connection_innov(i+1)
        c2 = p2.find_connection_innov(i+1)
        if c1 is None:
            return [c2, p2]
        if c2 is None:
            return [c1, p1]
        if not c1.enabled and c2.enabled:
            return [c1, p1]
        if not c2.enabled and c1.enabled:
            return [c2, p2]
        seed = np.random.random()
        if seed < 0.5:
            return [c1, p1]
        return [c2, p2]



    def crossover(self, p1, p2):
        child = NeuralNet_JIT(self.num_inputs, self.num_outputs, self.networks[0].master_connection_dict, self.networks[0].master_innov)
        for i in range(self.networks[0].master_innov[0] - 1):
            res = self.determine_connection(i, p1, p2)
            if res[0] is not None:
                connection = res[0]
                input_node = res[1].find_node(res[0].input_id)
                output_node = res[1].find_node(res[0].output_id)
                child.handle_new_connection(connection, input_node, output_node)
        child.revalidate_node_order()
        child.mutate(0.9)
        return child

    
    def top_pop(self, num):
        order = []
        nums = []
        for k in range(len(self.networks)):
            n = self.networks[k]
            nums.append(n.fitness)
        for i in range(num):
            max_index = nums.index(max(nums))
            nums[max_index] = -100000
            order.append(self.networks[max_index])
        return order

    def order_species(self, avg_fit):
        order = []
        nums = []
        for s in self.species:
            nums.append((self.avg_fitness(s) * len(s) / avg_fit))
        for i in range(len(nums)):
            max_index = nums.index(max(nums))
            nums[max_index] = -100000
            order.append(self.species[max_index])
        return order
    
    
    def prepare_next_gen(self, num_parents):
        avg_fit = self.avg_fitness(self.networks)
        new_pop = self.top_pop(num_parents)
        num_left = self.pop_size - num_parents
        species_list = self.order_species(avg_fit)
        counter = 0
        for s in species_list:
            if num_left > 0:
                counter+=1
                num_offspring = max(math.ceil(self.avg_fitness(s) * len(s) / avg_fit), 1)
                num_left -= (num_offspring)
                if num_left < 0:
                    num_offspring += num_left
                new_members = []
                for i in range(num_offspring):
                    print("Creating member " + str(len(new_pop) + i + 1) + "/" + str(self.pop_size) + "\t", end='\r')
                    parent1 = self.choose_parent(s)
                    parent2 = self.choose_parent(s)
                    new_members.append(NeuralNet_JIT.crossover(parent1, parent2, self.networks[0].master_connection_dict, self.networks[0].master_innov, self.num_inputs, self.num_outputs))
                if len(new_members) > 0:
                    new_pop.extend(new_members)
            else:
                break
        print("\n - Top " + str(counter) + " species survived\n", end='\n')
        self.networks = new_pop
        self.species = self.assign_pop_to_species(self.networks, [])


    def construct_non_jit(self, jit):
        child = NeuralNet(self.num_inputs, self.num_outputs)
        for i in range(self.networks[0].master_innov[0] - 1):
            res = self.determine_connection(i, jit, jit)
            child.handle_new_connection(res[0], res[1])
        child.revalidate_node_order()
        child.set_next_innov(self.networks[0].master_innov[0])
        return child

    def construct_jit(self, non_jit, mcd, master_innov):
        #master_innov[0] = non_jit.next_innov
        child = NeuralNet_JIT(self.num_inputs, self.num_outputs, mcd, master_innov)
        for i in range(non_jit.next_innov - 1):
            res = self.determine_connection(i, non_jit, non_jit)
            if res[0] is not None:
                con = res[0]
                inp = res[1].find_node(res[0].input_id)
                out = res[1].find_node(res[0].output_id)
                child.handle_new_connection(con.innov, con.input_id, con.output_id, con.enabled, con.weight, inp.bias, inp.type, out.bias, out.type)
        child.revalidate_node_order()
        return child

    def construct_jit_copy(self, jit, mcd, master_innov):
        child = NeuralNet_JIT(self.num_inputs, self.num_outputs, mcd, master_innov)
        for i in range(jit.master_innov[0] - 1):
            res = jit.find_connection_innov(i+1)
            if res is not None:
                con = res
                inp = jit.find_node(res.input_id)
                out = jit.find_node(res.output_id)
                child.handle_new_connection(con.innov, con.input_id, con.output_id, con.enabled, con.weight, inp.bias, inp.type, out.bias, out.type)
        child.revalidate_node_order()
        return child

    def construct_jit_connection(self, c):
        jit_c = Connection_JIT(c.input_id, c.output_id, c.innov)
        jit_c.weight = c.weight
        jit_c.enabled = c.enabled
        return jit_c

    def construct_jit_node(self, n):
        jit_n = Node_JIT(n.id, n.type, self.get_dummy_clist())
        jit_n.bias = n.bias
        jit_n.value = n.value
        for c in n.connections:
            jit_c = self.construct_jit_connection(c)
            jit_n.connections.append(jit_c)
        return jit_n

    def get_dummy_clist(self):
        c_list = numba.typed.List()
        c_list.append(Connection_JIT(0,0,0))
        return c_list



    def simulate_generations(self, num_generations, print_best):
        file_path = "csv/ESN_Results.csv"
        dw = DataWriter()
        dr = DataReader()
        dw.init_table(file_path)
        p = Predictor()
        mapping = dr.get_mapping()
        images = dr.get_images(112800, 28, 28) # 112800 images in data set
        scale_factor = 10
        for i in range(num_generations):
            sum = 0
            best_score = -100
            best_accuracy = -100
            best_net = []
            engines = []
            for net in self.networks:
                engine = [net, 0, 0]
                engines.append(engine)
            p.make_predictions(engines, mapping, images, scale_factor)
            for j in range(len(engines)):
                self.networks[j].fitness = engines[j][1]
                if engines[j][2] > best_score:
                    best_score = engines[j][2]
                    best_net = self.networks[j]
                if engines[j][1] > best_accuracy:
                    best_accuracy = engines[j][1]
            avg_accuracy = self.avg_fitness(self.networks) # avg accuracy
            for j in range(len(engines)):
                self.networks[j].fitness = engines[j][2] # change fitness to score
            avg_score = self.avg_fitness(self.networks) # avg accuracy
            avg_size = self.avg_network_size()
            if print_best:
                best_net.show_net()
            print("-----------------------------------\t\t\t\t\t\t\n       Generation " + str(i+1) + " results\n-----------------------------------\n", end='\n')
            print("Highest accuracy: " + str(best_accuracy*100) + "%\nHighest score: " + str(best_score**(1.0/scale_factor)) + "\nAverage accuracy: " + str(avg_accuracy*100) + "%\nAverage score: " + str(avg_score**(1.0/scale_factor)) + "\nNum species: " + str(len(self.species)) + "\nInnovs tried: " + str(self.networks[0].master_innov[0]) + "\nAverage connections per network: " + str(avg_size) + "\n")
            
            non_jit = self.construct_non_jit(best_net)
            pickle.dump(non_jit, open("neural_net.txt", "wb"))
            dw.write_row(file_path, [i+1, best_accuracy*100, avg_accuracy*100, best_score**(1.0/scale_factor), avg_score**(1.0/scale_factor), avg_size])
            if i != num_generations-1:
                self.prepare_next_gen(math.ceil(self.pop_size/10))
                print("\nStarting Generation " + str(i+2) + ": Species = " + str(len(self.species)) + ", Innovs = " + str(self.networks[0].master_innov[0]), end='\n')
        print("Finished simulation!")

    def calc_difference(self, net1, net2):
        tot_diff = 0
        weight_diff = 0
        num_matches = 0
        for c2 in net2.connection_list:
            con1 = net1.find_connection_innov(c2.innov)
            if con1 is not None:
                if con1.enabled and c2.enabled:
                    num_matches += 1
                    weight_diff += np.abs(con1.weight - c2.weight)
        tot_diff += (self.weight_scale * weight_diff / max(1, num_matches), 100) [num_matches == 0]
        num_connections = net1.num_active_connections() + net2.num_active_connections()
        tot_diff += self.struct_scale * (num_connections - 2 * num_matches) / num_connections
        return tot_diff

    def avg_network_size(self):
        sum = 0
        for net in self.networks:
            sum += net.num_active_connections()
        return sum/len(self.networks)

    def print_pop(self):
        print("Each child in the population: \n")
        for net in self.networks:
            net.show_net()
        print("\nAll species (" + str(len(self.species)) + "):\n")
        for i in range(len(self.species)):
            s = self.species[i]
            print("Species " + str(i+1) + "\n\n")
            for n in s:
                n.show_net()

        
        
    




