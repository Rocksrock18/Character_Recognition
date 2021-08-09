import numpy as np
from Node import Node
from Connection import Connection
import copy as copy
from random import shuffle

class NeuralNet():
    def __init__(self, num_inputs, num_outputs):
        self.node_list = []
        self.node_dict = {}
        self.connection_dict = {}
        self.connection_list = []
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.node_order = []
        self.node_order_dict = dict() # {key: node_id, value: node_order_index}
        self.node_order_index_dict = dict() # {key: node_order_index, value: node_id}
        self.next_id = 1
        self.fitness = 0
        self.next_innov = 0

    def set_next_innov(self, innov):
        self.next_innov = innov

    #will return value between 0 and 1.
    def sigmoid(self, x):
        #boundary check to get rid of potential overflow. Little effect on training
        if x > 100:
            return .99999
        elif x < -100:
            return 0.00001
        return 1/(1 + np.exp(-x))


    def relu(self, x):
        return (x, 0) [x < 0]


    
    def determine_next_id(self):
        start = self.input_size + self.output_size
        for i in range(start, 1000000):
            if self.find_node(i) is None:
                self.next_id = i
                return


    def master_reset_innov(self):
        new_innov = 1
        for c in self.connection_list:
            old_innov = c.innov
            self.connection_dict.pop(old_innov)
            c.innov = new_innov
            new_innov += 1
            self.connection_dict[c.innov] = c
        self.next_innov = len(self.connection_list)
    
    
    def get_placement_index(self, start, stop, step):
        #print("Start = " + str(start))
        for i in range(start, stop, step):
            if step == -1:
                if self.find_node(self.node_order_index_dict[i]).type != "Output": # self.node_order[i]
                    return i+1
            else:
                if self.find_node(self.node_order_index_dict[i]).type != "Input": # self.node_order[i]
                    return i
        return len(self.node_order_dict)
    
    def reset_node_values(self, inputs):
        next_input = 0
        for node in self.node_list:
            if node.type == "Input":
                node.value = inputs[next_input]
                next_input += 1
            else:
                node.value = 0


    #adds and sorts nodes to maintain proper node order
    def add_and_sort_nodes2(self, input_node, output_node):
        #temp = set(self.node_order)
        #print(input_node)
        #print(output_node)
        if input_node in self.node_order and output_node in self.node_order:
            if self.node_order.index(input_node) > self.node_order.index(output_node):
                self.node_order.remove(input_node)
                self.node_order.insert(self.node_order.index(output_node), input_node)
        elif input_node in self.node_order:
            self.node_order.insert(self.get_placement_index(self.node_order.index(input_node)+1, len(self.node_order), 1), output_node)
        elif output_node in self.node_order:
            #print("Index = " + str(self.get_placement_index(self.node_order.index(output_node), -1, -1)))
            self.node_order.insert(self.get_placement_index(self.node_order.index(output_node), -1, -1), input_node)
            #print(self.node_order)
        else:
            self.node_order.append(input_node)
            self.node_order.append(output_node)

    def add_and_sort_nodes(self, input_node, output_node):
        #print("Before\n")
        #print(self.node_order_dict)
        #print(self.node_order_index_dict)
        #print("Input = " + str(input_node))
        #print("Output = " + str(output_node))
        if input_node in self.node_order_dict and output_node in self.node_order_dict:
            if self.node_order_dict[input_node] > self.node_order_dict[output_node]:
                self.trickle_down_node_order(self.node_order_dict[input_node], self.node_order_dict[output_node], input_node)
        elif input_node in self.node_order_dict:
            index = self.get_placement_index(self.node_order_dict[input_node]+1, len(self.node_order_dict)+1, 1)
            #print(index)
            self.trickle_down_node_order(len(self.node_order_index_dict)+1, index, output_node)
        elif output_node in self.node_order_dict:
            index = self.get_placement_index(self.node_order_dict[output_node], -1, -1)
            self.trickle_down_node_order(len(self.node_order_index_dict)+1, index, input_node)
        else:
            self.node_order_index_dict[len(self.node_order_index_dict)+1] = input_node
            self.node_order_dict[input_node] = len(self.node_order_dict)+1
            self.node_order_index_dict[len(self.node_order_index_dict)+1] = output_node
            self.node_order_dict[output_node] = len(self.node_order_dict)+1
        #print("After\n")
        #print(self.node_order_dict)
        #print(self.node_order_index_dict)
        #input()
    
    # start is index that node used to be in, stop is the new index of the node
    def trickle_down_node_order(self, start, stop, node):
        #print("Trickle down\n")
        for i in range(start, stop, -1):
            #print(i)
            self.node_order_index_dict[i] = self.node_order_index_dict[i-1]
            self.node_order_dict[self.node_order_index_dict[i]] = i
            #print(self.node_order_dict)
            #print(self.node_order_index_dict)
        self.node_order_index_dict[stop] = node
        self.node_order_dict[node] = stop

    def trickle_up_node_order(self, start, stop):
        #print("Trickle up\n")
        for i in range(start, stop):
            #print(i)
            self.node_order_index_dict[i] = self.node_order_index_dict[i+1]
            self.node_order_dict[self.node_order_index_dict[i]] = i
            #print(self.node_order_dict)
            #print(self.node_order_index_dict)
        self.node_order_index_dict.pop(stop)

    # randomizes all bias values
    def randomize_all_bias(self):
        for n in self.node_list:
            n.bias = (np.random.randn(), 0) [n.type == "Input"]
    

    def index_of_connection(self, c, c_list):
        for i in range(len(c_list)):
            con = c_list[i]
            if con.input_id == c.input_id and con.output_id == c.output_id:
                return i
        return -1

    def index_of_output_connection(self, c, c_list):
        for i in range(len(c_list)):
            con = c_list[i]
            if con.output_id == c.output_id:
                return i
        return -1


    def index_of_node(self, n, n_list):
        for i in range(len(n_list)):
            node = n_list[i]
            if n.id == node.id:
                return i
        return -1

    def num_nodes_connected_to(self, node_id):
        num_connected = 0
        for c in self.connection_list:
            if c.output_id == node_id:
                num_connected += 1
        return num_connected
    
    def connection_can_be_deleted_old(self, input_id, output_id):
        input_node = self.find_node(input_id)
        output_node = self.find_node(output_id)
        if output_node.type != "Hidden":
            return False
        to_be_connected = self.retrieve_enabled_connections(output_node)
        return self.node_connected_to_all(input_node, to_be_connected)

    def connection_can_be_deleted(self, output_id):
        output_node = self.find_node(output_id)
        return output_node.type == "Hidden"

    
    def retrieve_enabled_connections(self, node):
        enabled = []
        for c in node.connections:
            if c.enabled:
                enabled.append(c)
        return enabled
    
    
    def node_connected_to_all(self, node, connections):
        for c in node.connections:
            if c.enabled:
                index = self.index_of_output_connection(c, connections)
                if index != -1:
                    connections.pop(index)
        return len(connections) == 0

    def delete_connection(self):
        c_list = copy.deepcopy(self.connection_list)
        shuffle(c_list)
        for c in c_list:
            if c.enabled and self.connection_can_be_deleted(c.output_id):
                #print("\nDeleting connection from " + str(c.input_id) + " and " + str(c.output_id))
                #self.connection_list.pop(self.index_of_connection(c, self.connection_list))
                #self.connection_dict.pop(c.innov)
                #node = self.find_node(c.input_id)
                #node.connections.pop(self.index_of_connection(c, node.connections))
                self.unassociate(self.find_node(c.input_id), self.retrieve_enabled_connections(self.find_node(c.output_id)), c)
                if self.num_nodes_connected_to(c.output_id) == 0:
                    self.remove_node(self.find_node(c.output_id))
                return

    def unassociate(self, input_node, new_connections, old_connection):
        #print("\n")
        #print(self.index_of_connection(old_connection, input_node.connections))
        #print(self.index_of_connection(old_connection, self.connection_list))
        input_node.connections.pop(self.index_of_connection(old_connection, input_node.connections))
        self.connection_list.pop(self.index_of_connection(old_connection, self.connection_list))
        self.connection_dict.pop(old_connection.innov)
        #self.validate_connections()
        for new_c in new_connections:
            output_node = self.find_node(new_c.output_id)
            if self.connection_valid(input_node, output_node):
                self.append_connection(input_node, output_node, old_connection.weight * new_c.weight) # not exactly correct
            elif self.connection_exists(input_node, output_node):
                c = self.search_connections(input_node.id, output_node.id)
                if c.enabled:
                    c.weight += old_connection.weight * new_c.weight
                else:
                    c.weight = old_connection.weight * new_c.weight
                c.enabled = True



    def remove_node(self, node):
        to_be_connected = self.retrieve_enabled_connections(node)
        for n in self.node_list:
            for c in n.connections:
                if c.output_id == node.id and c.input_id == n.id:
                    self.unassociate(n, to_be_connected, c)
                    break
        for c in node.connections:
            self.connection_list.pop(self.index_of_connection(c, self.connection_list))
            self.connection_dict.pop(c.innov)
        self.node_list.pop(self.index_of_node(node, self.node_list))
        self.node_dict.pop(node.id)
        #self.node_order.remove(node.id)
        index = self.node_order_dict.pop(node.id)
        self.node_order_index_dict.pop(index)
        self.trickle_up_node_order(index, len(self.node_order_index_dict)+1)

    def delete_node(self):
        n_list = copy.deepcopy(self.node_list)
        shuffle(n_list)
        for n in n_list:
            if n.type == "Hidden":
                #print("\nDeleting Node " + str(n.id))
                #self.show_net()
                self.remove_node(n)
                #self.show_net()
                #input("CHECK")
                return
    
    def num_active_connections(self):
        sum = 0
        for c in self.connection_list:
            if c.enabled:
                sum += 1
        return sum


    def connection_exists(self, n1, n2):
        for c in self.connection_list:
            if c.input_id == n1.id and c.output_id == n2.id:
                    return True
        return False


    def creates_circular_dependency(self, target_id, n2):
        for c in n2.connections:
            output_node = self.find_node(c.output_id)
            if output_node.id == target_id:
                return True
            if self.creates_circular_dependency(target_id, output_node):
                return True
        return False


    def new_connection_valid(self, n1, n2):
        if n1 is None or n2 is None:
            return True
        if n1.id == n2.id:
            return False
        if n1.type == "Hidden" and self.creates_circular_dependency(n1.id, n2):
            return False
        if n1.type == "Input":
            if n2.type == "Hidden" or n2.type == "Output":
                return True
        elif n1.type == "Hidden":
            if n2.type == "Hidden" or n2.type == "Output":
                return True
        return False


    def connection_valid(self, n1, n2):
        if n1 is None or n2 is None:
            return True
        if n1.id == n2.id:
            return False
        if self.connection_exists(n1, n2):
            return False
        if self.connection_exists(n2, n1):
            return False
        if self.node_order_dict[n1.id] > self.node_order_dict[n2.id]: # self.node_order.index()
            return False
        if n1.type == "Input":
            if n2.type == "Hidden" or n2.type == "Output":
                return True
        elif n1.type == "Hidden":
            if n2.type == "Hidden" or n2.type == "Output":
                return True
        return False

    
    
    def revalidate_node_order(self):
        order_changed = True
        #print("Start")
        while order_changed:
            order_changed = False
            for c in self.connection_list:
                if self.find_node(c.input_id).type == "Hidden" and self.find_node(c.output_id).type == "Hidden":
                    if self.node_order_dict[c.input_id] > self.node_order_dict[c.output_id]: # self.node_order.index()
                        self.add_and_sort_nodes(c.input_id, c.output_id)
                        #print("Do it again")
                        order_changed = True
                        break


    
    def copy_bias(self, parent):
        for n in self.node_list:
            pn = parent.find_node(n.id)
            if not pn is None:
                n.bias = pn.bias



    def get_valid_outputs(self):
        valid = []
        for node in self.node_list:
            if node.type != "Input":
                valid.append(node)
        return copy.deepcopy(valid)


    def get_valid_inputs(self):
        valid = []
        for node in self.node_list:
            if node.type != "Output":
                valid.append(node)
        return copy.deepcopy(valid)


    def add_connection(self):
        #print("Add connection")
        n_list1 = self.get_valid_inputs()
        n_list2 = self.get_valid_outputs()
        shuffle(n_list1)
        shuffle(n_list2)
        #print("Lists shuffled")
        for n1 in n_list1:
            for n2 in n_list2:
                if self.connection_valid(n1, n2):
                    real_n1 = self.find_node(n1.id)
                    real_n2 = self.find_node(n2.id)
                    self.append_connection(real_n1, real_n2, 0)
                    return
            #print("Next node")
        return



    # completely randomize weights of network
    def randomize(self):
        for c in self.connection_list:
            c.weight = np.random.randn()
        for n in self.node_list:
            n.bias = (np.random.randn(), 0) [n.type == "Input"]
    
    def get_output_nodes(self):
        nodes = []
        for n in self.node_list:
            if n.type == "Output":
                nodes.append(n.value)
        return nodes

    def get_input_nodes(self):
        nodes = []
        for n in self.node_list:
            if n.type == "Input":
                nodes.append(n)
        return nodes
    
    
    def search_connections(self, input, output):
        for c in self.connection_list:
            if c.input_id == input and c.output_id == output:
                return c
        return
    
    
    def determine_innov(self, input, output):
        c = self.data.get_existing_connection(input, output)
        if c is None:
            innov = self.data.get_innov()
            return innov
        return c.innov

    
    def handle_new_connection(self, c, parent):
        if c is None:
            return
        existing_c = self.find_connection_innov(c.innov)
        input_bias = parent.find_node(c.input_id).bias
        output_bias = parent.find_node(c.output_id).bias
        if existing_c is None:
            new_node_input = self.find_node(c.input_id)
            new_node_output = self.find_node(c.output_id)
            if not self.new_connection_valid(new_node_input, new_node_output):
                return
            if new_node_input is None:
                new_node_input = Node(c.input_id, parent.find_node(c.input_id).type)
                new_node_input.bias = input_bias
                self.node_list.append(new_node_input)
                self.node_dict[new_node_input.id] = new_node_input
                self.determine_next_id()
            if new_node_output is None:
                new_node_output = Node(c.output_id, parent.find_node(c.output_id).type)
                new_node_output.bias = output_bias
                self.node_list.append(new_node_output)
                self.node_dict[new_node_output.id] = new_node_output
                self.determine_next_id()
            self.add_and_sort_nodes(new_node_input.id, new_node_output.id)
            new_c = Connection(new_node_input.id, new_node_output.id, c.innov)
            new_c.enabled = c.enabled
            new_c.weight = c.weight
            self.connection_list.append(new_c)
            self.connection_dict[new_c.innov] = new_c
            new_node_input.connections.append(new_c)
        else:
            existing_c.enabled = c.enabled
            existing_c.weight = c.weight

    def show_connection_list(self):
        for c in self.connection_list:
            print(str(c))
    
    def find_connection_innov_old(self, innov):
        for c in self.connection_list:
            if c.innov == innov:
                return c
        return

    def find_connection_innov(self, innov):
        if innov in self.connection_dict:
            return self.connection_dict[innov]
    
    
    def find_node_old(self, id):
        for n in self.node_list:
            if n.id == id:
                return n
        return

    def find_node(self, id):
        if id in self.node_dict:
            return self.node_dict[id]
    
    # assumes that node list has all inputs first. dangerous assumption, shouldnt use
    def init_inputs(self, inputs):
        for i in range(len(inputs)): 
            self.node_list[i].value = inputs[i]
   

    def show_net(self):
        print("", end='\n')
        for i in range(1, self.next_id):
            n = self.find_node(i)
            if n is not None:
                print(str(n))
        print("\n")


    def show_connection_dict(self):
        for key in self.connection_dict:
            print(str(key) + ": " + str(self.connection_dict[key]))
        print("\n")



