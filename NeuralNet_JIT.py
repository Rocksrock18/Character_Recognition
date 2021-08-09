import numpy as np
from Node_JIT import Node_JIT
from Connection_JIT import Connection_JIT
import numba

list_instance = numba.typed.List()
list_instance.append(Connection_JIT(0,0,0))

list_instance2 = numba.typed.List()
list_instance2.append(Connection_JIT(0,0,0))

node_type = Node_JIT(0, "Input", list_instance, list_instance2)
connection_type = Connection_JIT(0,0,0)

n_list_type = numba.typed.List()
n_list_type.append(node_type)
c_list_type = numba.typed.List()
c_list_type.append(connection_type)

kv_ty1 = (numba.int64, numba.typeof(node_type))
kv_ty2 = (numba.int64, numba.typeof(connection_type))
kv_ty3 = (numba.int64, numba.typeof(c_list_type))
kv_ty4 = (numba.int64, numba.int64)
kv_ty5 = (numba.types.containers.UniTuple(numba.int64, 2), numba.typeof(connection_type))


spec = [
    ('node_list', numba.typeof(n_list_type)),
    ('node_dict', numba.types.DictType(*kv_ty1)),
    ('node_connection_dict', numba.types.DictType(*kv_ty3)),
    ('connection_dict', numba.types.DictType(*kv_ty2)),
    ('master_connection_dict', numba.types.DictType(*kv_ty5)),
    ('connection_list', numba.typeof(c_list_type)),
    ('input_size', numba.int64),
    ('output_size', numba.int64),
    ('node_order', numba.types.ListType(numba.int64)),
    ('node_order_dict', numba.types.DictType(*kv_ty4)),
    ('node_order_index_dict', numba.types.DictType(*kv_ty4)),
    ('next_id', numba.int64),
    ('fitness', numba.f8),
    ('master_innov', numba.types.ListType(numba.int64)),
]

@numba.experimental.jitclass(spec)
class NeuralNet_JIT(object):
    def __init__(self, num_inputs, num_outputs, mcd, next_innov):
        self.node_list = self.get_dummy_nlist()
        self.node_list.pop(0)
        self.node_dict = numba.typed.Dict.empty(*kv_ty1)
        self.node_connection_dict = numba.typed.Dict.empty(*kv_ty3)
        self.connection_dict = numba.typed.Dict.empty(*kv_ty2)
        self.master_connection_dict = mcd
        self.connection_list = self.get_dummy_clist()
        self.connection_list.pop(0)
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.node_order = numba.typed.List.empty_list(numba.types.int64)
        self.node_order_dict = numba.typed.Dict.empty(*kv_ty4) # {key: node_id, value: node_order_index}
        self.node_order_index_dict = numba.typed.Dict.empty(*kv_ty4) # {key: node_order_index, value: node_id}
        self.next_id = 1
        self.fitness = 0.0
        self.master_innov = next_innov

    def get_dummy_clist(self):
        c_list = numba.typed.List()
        c_list.append(Connection_JIT(0,0,0))
        return c_list

    def get_dummy_nlist(self):
        c_list = self.get_dummy_clist()
        n_list = numba.typed.List()
        n_list.append(Node_JIT(0, "Dummy", c_list, self.get_dummy_clist()))
        return n_list

    def reset_neural_net(self):
        self.next_id = 1
        for i in range(self.input_size):
            n = Node_JIT(self.next_id, "Input", self.get_dummy_list(), self.get_dummy_clist())
            self.node_list.append(n)
            self.node_dict[n.id] = n
            self.next_id += 1
        for k in range(self.output_size):
            new_node = Node_JIT(self.next_id, "Output", self.get_dummy_list(), self.get_dummy_clist())
            self.node_list.append(new_node)
            self.node_dict[new_node.id] = new_node
            for ii in range(self.input_size):
                c = Connection_JIT(ii+1, self.next_id, k*self.input_size + ii+1)
                if ii+1 in self.node_connection_dict:
                    current_list = self.node_connection_dict[ii+1]
                    current_list.append(c)
                    self.node_connection_dict[ii+1] = current_list
                else:
                    temp_list = numba.typed.List()
                    temp_list.append(c)
                    self.node_connection_dict[ii+1] = temp_list
                self.connection_list.append(c)
                self.connection_dict[c.innov] = c
                node = self.find_node(ii+1)
                node.connections.append(c)
                new_node.receptions.append(c)
                self.add_and_sort_nodes(ii+1, new_node.id)
                self.add_master_connection(c)
            self.next_id += 1
        return self.connection_list

    def add_master_connection(self, c):
        if (c.input_id, c.output_id) in self.master_connection_dict:
            return
        self.master_connection_dict[(c.input_id, c.output_id)] = c
        self.master_innov[0] = self.master_innov[0] + 1

    def get_dummy_list(self):
        new_list = numba.typed.List()
        new_list.append(Connection_JIT(0,0,0))
        return new_list


    def calc_affinity_input(self, c, net, images_answers, images_pixels, output_list, input_node): #, net, images_answers, images_pixels, output_list, input_node
        affinity = 0
        last_input = 784
        for i in range(len(images_pixels)):
            image = images_pixels[i]
            correct = images_answers[i]
            for o in output_list:
                if o == 0:
                    break
                #affinity = affinity + inputs[c.input_id - 1] * 46 if (o-last_input-1) == correct else affinity - inputs[c.input_id-1]
                if (o-last_input-1) == correct:
                    affinity += image[c.input_id - 1] * 46
                else:
                    affinity -= image[c.input_id - 1]
        return affinity/len(images_pixels)


    def add_and_sort_nodes(self, input_node, output_node):
        if input_node in self.node_order_dict and output_node in self.node_order_dict:
            if self.node_order_dict[input_node] > self.node_order_dict[output_node]:
                self.trickle_down_node_order(self.node_order_dict[input_node], self.node_order_dict[output_node], input_node)
        elif input_node in self.node_order_dict:
            index = self.get_placement_index(self.node_order_dict[input_node]+1, len(self.node_order_dict)+1, 1)
            self.trickle_down_node_order(len(self.node_order_index_dict)+1, index, output_node)
        elif output_node in self.node_order_dict:
            index = self.get_placement_index(self.node_order_dict[output_node], -1, -1)
            self.trickle_down_node_order(len(self.node_order_index_dict)+1, index, input_node)
        else:
            self.node_order_index_dict[len(self.node_order_index_dict)+1] = input_node
            self.node_order_dict[input_node] = len(self.node_order_dict)+1
            self.node_order_index_dict[len(self.node_order_index_dict)+1] = output_node
            self.node_order_dict[output_node] = len(self.node_order_dict)+1


    # start is index that node used to be in, stop is the new index of the node
    def trickle_down_node_order(self, start, stop, node):
        for i in range(start, stop, -1):
            self.node_order_index_dict[i] = self.node_order_index_dict[i-1]
            self.node_order_dict[self.node_order_index_dict[i]] = i
        self.node_order_index_dict[stop] = node
        self.node_order_dict[node] = stop

    def trickle_up_node_order(self, start, stop):
        for i in range(start, stop):
            self.node_order_index_dict[i] = self.node_order_index_dict[i+1]
            self.node_order_dict[self.node_order_index_dict[i]] = i
        self.node_order_index_dict.pop(stop)

    def get_placement_index(self, start, stop, step):
        for i in range(start, stop, step):
            if step == -1:
                if self.find_node(self.node_order_index_dict[i]).type != "Output": # self.node_order[i]
                    return i+1
            else:
                if self.find_node(self.node_order_index_dict[i]).type != "Input": # self.node_order[i]
                    return i
        return len(self.node_order_dict)

    def determine_next_id(self):
        start = self.input_size + self.output_size
        for i in range(start, 1000000):
            if self.find_node(i) is None:
                self.next_id = i
                return

    def determine_innov(self, input, output):
        c = self.get_existing_connection(input, output)
        if c is None:
            return self.master_innov[0]
        return c.innov

    def get_existing_connection(self, input, output):
        if (input, output) in self.master_connection_dict:
            return self.master_connection_dict[(input, output)]

    def master_reset_innov(self):
        for c in self.connection_list:
            old_innov = c.innov
            self.connection_dict.pop(old_innov)
            new_innov = self.determine_innov(c.input_id, c.output_id)
            c.innov = new_innov
            self.connection_dict[c.innov] = c
            self.add_master_connection(c)


    def test_matmul(self, m1, m2):
        return np.matmul(m1, m2)

    def reset_node_values(self, inputs):
        next_input = 0
        for i in range(len(self.node_list)):
            node = self.node_list[i]
            if node.type == "Input":
                node.value = inputs[next_input]
                next_input += 1
            else:
                node.value = 0
            #node.correction = 0.0

    #takes input info and propogates forward through the nn.
    def forward_prop(self, inputs):
        self.reset_node_values(inputs)
        for id in range(1, len(self.node_order_index_dict)+1): # self.node_order
            node = self.find_node(self.node_order_index_dict[id]) #self.find_node(id)
            #print(str(node.id) + " = " + node.float_to_string(node.value) + ", bias = " + node.float_to_string(node.bias))
            node.value += node.bias
            #print("Node: " + node.to_string())
            if node.type != "Input": # was not input w/ sigmoid. with relu, make sure its hidden
                node.value = self.sigmoid(node.value) # can use relu or sigmoid
            if node.value != 0: # node.value != 0 with relu
                for c in node.connections:
                    if c.enabled:
                        output = self.find_node(c.output_id)
                        output.value += node.value * c.weight
        return self.get_output_node_values()

    def back_prop(self):
        for id in range(len(self.node_order_index_dict), 0, -1):
            node = self.find_node(self.node_order_index_dict[id])
            if node.correction != 0:
                if node.type != "Input":
                    node.bias += node.correction * .00001
                    if np.sign(node.correction) == np.sign(node.affinity):
                        node.bias += node.affinity * .001
                for c in node.receptions:
                    input = self.find_node(c.input_id)
                    correction = input.value * node.correction * .0001
                    if np.sign(correction) == np.sign(c.affinity):
                        c.weight += c.affinity * .001
                    c.weight += correction
                    input.correction += correction

    def get_output_node_values(self):
        nodes = numba.typed.List.empty_list(numba.f8)
        for i in range(len(self.node_list)):
            n = self.node_list[i]
            if n.type == "Output":
                nodes.append(n.value)
        return nodes

    def get_output_nodes(self):
        nodes = self.get_dummy_nlist()
        nodes.pop(0)
        for i in range(len(self.node_list)):
            n = self.node_list[i]
            if n.type == "Output":
                nodes.append(n)
        return nodes

    # randomizes all bias values
    def randomize_all_bias(self):
        for n in self.node_list:
            if n.type != "Input":
                n.bias = np.random.randn()


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


    def creates_circular_dependency(self, target_id, n2):
        nodes_to_check = self.get_dummy_nlist()
        nodes_to_check.pop(0)
        nodes_to_check.append(n2)
        while len(nodes_to_check) > 0:
            node = nodes_to_check[0]
            for c in node.connections:
                output_node = self.find_node(c.output_id)
                if output_node.id == target_id:
                    return True
                nodes_to_check.append(output_node)
            nodes_to_check.pop(0)
        return False

    @staticmethod
    def calc_difference(net1, net2, weight_scale, struct_scale):
        tot_diff = 0
        weight_diff = 0
        num_matches = 0
        for c2 in net2.connection_list:
            con1 = net1.find_connection_innov(c2.innov)
            if con1 is not None:
                if con1.enabled and c2.enabled:
                    num_matches += 1
                    weight_diff += np.abs(con1.weight - c2.weight)
        if num_matches == 0:
            tot_diff = 100
        else:
            tot_diff = weight_scale * weight_diff / num_matches
        num_connections = net1.num_active_connections() + net2.num_active_connections()
        tot_diff += struct_scale * (num_connections - 2 * num_matches) / num_connections
        return tot_diff


    def handle_new_connection(self, innov, input_id, output_id, enabled, weight, in_bias, in_type, out_bias, out_type):
        #if c is None:
        #    return
        existing_c = self.find_connection_innov(innov)
        input_bias = in_bias
        output_bias = out_bias
        if existing_c is None:
            new_node_input = self.find_node(input_id)
            new_node_output = self.find_node(output_id)
            if not self.new_connection_valid(new_node_input, new_node_output):
                return
            if new_node_input is None:
                new_node_input = Node_JIT(input_id, in_type, self.get_dummy_clist(), self.get_dummy_clist())
                new_node_input.bias = input_bias
                self.node_list.append(new_node_input)
                self.node_dict[new_node_input.id] = new_node_input
                self.determine_next_id()
            if new_node_output is None:
                new_node_output = Node_JIT(output_id, out_type, self.get_dummy_clist(), self.get_dummy_clist())
                new_node_output.bias = output_bias
                self.node_list.append(new_node_output)
                self.node_dict[new_node_output.id] = new_node_output
                self.determine_next_id()
            self.add_and_sort_nodes(new_node_input.id, new_node_output.id)
            new_c = Connection_JIT(new_node_input.id, new_node_output.id, innov)
            new_c.enabled = enabled
            new_c.weight = weight
            self.connection_list.append(new_c)
            self.connection_dict[new_c.innov] = new_c
            self.add_master_connection(new_c)
            new_node_input.connections.append(new_c)
            new_node_output.receptions.append(new_c)
        else:
            existing_c.enabled = enabled
            existing_c.weight = weight

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

    # FINISH ADDING MUTATION FUNCTIONALITY

    # mutates the neural net
    def mutate(self, weight_rate):
        if np.random.random() < 0.01:
            self.add_node()
        if np.random.random() < 0.01:
            self.delete_node()
        if np.random.random() < 0.03:
            self.add_connection()
        if np.random.random() < 0.03:
            self.delete_connection()
        if np.random.random() < weight_rate:
            self.mutate_weights()
        return

    def add_node(self):
        con = self.connection_list[np.random.randint(0, len(self.connection_list))]
        input_n = self.find_node(con.input_id)
        output_n = self.find_node(con.output_id)
        con.enabled = False
        new_node = Node_JIT(self.next_id, "Hidden", self.get_dummy_list(), self.get_dummy_clist())
        new_node.bias = 0 # makes it so the new node doesnt change the network, just changes the possibilities
        self.node_list.append(new_node)
        self.node_dict[new_node.id] = new_node
        self.determine_next_id()

        c1 = Connection_JIT(con.input_id, new_node.id, self.determine_innov(con.input_id, new_node.id))
        c1.weight = con.weight # gets weight of old connection
        self.connection_list.append(c1)
        self.connection_dict[c1.innov] = c1
        input_n.connections.append(c1)
        new_node.receptions.append(c1)
        self.add_master_connection(c1)
        self.add_and_sort_nodes(input_n.id, new_node.id)

        c2 = Connection_JIT(new_node.id, con.output_id, self.determine_innov(new_node.id, con.output_id))
        c2.weight = 1 # makes it so the new node doesnt change the network, just changes the possibilities
        self.connection_list.append(c2)
        self.connection_dict[c2.innov] = c2
        new_node.connections.append(c2)
        output_n.receptions.append(c2)
        self.add_master_connection(c2)
        self.add_and_sort_nodes(new_node.id, output_n.id)

    # success
    def delete_node(self):
        n_list = self.get_node_ids()
        np.random.shuffle(n_list)
        for i in range(len(n_list)):
            n = self.find_node(n_list[i])
            if n.type == "Hidden":
                self.remove_node(n)
                return

    def get_node_ids(self):
        ids = np.arange(0)
        for i in range(len(self.node_list)):
            node = self.node_list[i]
            ids = np.append(ids, node.id)
        return ids

    # success
    def remove_node(self, node):
        to_be_connected = self.retrieve_enabled_connections(node)
        for i in range(len(self.node_list)):
            n = self.node_list[i]
            for c in n.connections:
                if c.output_id == node.id and c.input_id == n.id:
                    self.unassociate(n, to_be_connected, c)
                    break
        for c in node.connections:
            self.connection_list.pop(self.index_of_connection(c, self.connection_list))
            self.connection_dict.pop(c.innov)
        self.node_list.pop(self.index_of_node(node, self.node_list))
        self.node_dict.pop(node.id)
        index = self.node_order_dict.pop(node.id)
        self.node_order_index_dict.pop(index)
        self.trickle_up_node_order(index, len(self.node_order_index_dict)+1)

    # success
    def unassociate(self, input_node, new_connections, old_connection):
        output_node = self.find_node(old_connection.output_id)
        input_node.connections.pop(self.index_of_connection(old_connection, input_node.connections))
        output_node.receptions.pop(self.index_of_connection(old_connection, output_node.receptions))
        self.connection_list.pop(self.index_of_connection(old_connection, self.connection_list))
        self.connection_dict.pop(old_connection.innov)
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

    def add_connection(self):
        n_list1 = self.get_valid_inputs()
        n_list2 = self.get_valid_outputs()
        np.random.shuffle(n_list1)
        np.random.shuffle(n_list2)
        for i1 in range(len(n_list1)):
            for i2 in range(len(n_list2)):
                if self.connection_valid(self.find_node(n_list1[i1]), self.find_node(n_list2[i2])):
                    real_n1 = self.find_node(n_list1[i1])
                    real_n2 = self.find_node(n_list2[i2])
                    self.append_connection(real_n1, real_n2, 0)
                    return
        return

    # success
    def append_connection(self, n1, n2, default_weight = None):
        index1 = self.node_order_dict[n1.id] # self.node_order.index()
        index2 = self.node_order_dict[n2.id]
        if index1 < index2:
            c = Connection_JIT(n1.id, n2.id, self.determine_innov(n1.id, n2.id))
            if default_weight is not None:
                c.weight = default_weight
            self.connection_list.append(c)
            self.connection_dict[c.innov] = c
            n1.connections.append(c)
            n2.receptions.append(c)
            self.add_master_connection(c)
            self.add_and_sort_nodes(n1.id, n2.id)
        else:
            c = Connection_JIT(n2.id, n1.id, self.determine_innov(n2.id, n1.id))
            if default_weight is not None:
                c.weight = default_weight
            self.connection_list.append(c)
            self.connection_dict[c.innov] = c
            n2.connections.append(c)
            n1.receptions.append(c)
            self.add_master_connection(c)
            self.add_and_sort_nodes(n2.id, n1.id)

    def delete_connection(self):
        c_list = self.get_connection_innovs()
        np.random.shuffle(c_list)
        for i in range(len(c_list)):
            c = self.find_connection_innov(c_list[i])
            if c.enabled and self.connection_can_be_deleted(c.output_id):
                self.unassociate(self.find_node(c.input_id), self.retrieve_enabled_connections(self.find_node(c.output_id)), c)
                if len(self.find_node(c.output_id).receptions) == 0 or len(self.find_node(c.output_id).connections) == 0:
                    self.remove_node(self.find_node(c.output_id))
                return

    def get_connection_innovs(self):
        innovs = np.arange(0)
        for c in self.connection_list:
            innovs = np.append(innovs, c.innov)
        return innovs

    def mutate_weights(self):
        for n in self.node_list:
            if n.type != "Input":
                if np.random.random() < 0.02:
                    n.bias = np.random.randn()
                elif np.random.random() < 0.20:
                    n.bias += 2*np.random.random() - 1
        for c in self.connection_list:
            if c.enabled:
                if np.random.random() < 0.02:
                    c.weight = np.random.randn()
                elif np.random.random() < 0.20:
                    c.weight += 2*np.random.random() - 1
        return

    def retrieve_enabled_connections(self, node):
        enabled = numba.typed.List()
        for c in node.connections:
            if c.enabled:
                enabled.append(c)
        return enabled

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


    def determine_connection(self, i, c1, c2):
        if not c1.enabled and c2.enabled:
            return 0
        if not c2.enabled and c1.enabled:
            return 1
        seed = np.random.random()
        if seed < 0.5:
            return 0
        return 1

    @staticmethod
    def crossover(p1, p2, mcd, mi, num_inputs, num_outputs):
        child = NeuralNet_JIT(num_inputs, num_outputs, mcd, mi)
        for i in range(mi[0] - 1):
            if (i+1) in p1.connection_dict:
                c1 = p1.connection_dict[i+1]
                if (i+1) in p2.connection_dict:
                    c2 = p2.connection_dict[i+1]
                    res = child.determine_connection(i, c1, c2)
                else:
                    res = 0
            elif (i+1) in p2.connection_dict:
                c2 = p2.connection_dict[i+1]
                res = 1
            else:
                res = 2
            if res == 0:
                if c1 is not None:
                    con = c1
                    inp = p1.find_node(c1.input_id)
                    out = p1.find_node(c1.output_id)
                    child.handle_new_connection(con.innov, con.input_id, con.output_id, con.enabled, con.weight, inp.bias, inp.type, out.bias, out.type)
            elif res == 1:
                if c2 is not None:
                    con = c2
                    inp = p2.find_node(c2.input_id)
                    out = p2.find_node(c2.output_id)
                    child.handle_new_connection(con.innov, con.input_id, con.output_id, con.enabled, con.weight, inp.bias, inp.type, out.bias, out.type)
        child.revalidate_node_order()
        child.mutate(1)
        return child

    #will return value between 0 and 1.
    def sigmoid(self, x):
        #boundary check to get rid of potential overflow. Little effect on training
        if x > 50:
            return 1
        elif x < -50:
            return 0
        return np.divide(1, 1 + np.exp(-x))


    def relu(self, x):
        return x if x > 0 else 0

    def find_node(self, id):
        if id in self.node_dict:
            return self.node_dict[id]

    def find_connection_innov(self, innov):
        if innov in self.connection_dict:
            return self.connection_dict[innov]


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

    def search_connections(self, input, output):
        for c in self.connection_list:
            if c.input_id == input and c.output_id == output:
                return c
        return

    def connection_exists(self, n1, n2):
        for c in self.connection_list:
            if c.input_id == n1.id and c.output_id == n2.id:
                    return True
        return False

    def connection_can_be_deleted(self, output_id):
        output_node = self.find_node(output_id)
        return output_node.type == "Hidden"

    def get_valid_outputs(self):
        valid = np.arange(0)
        for node in self.node_list:
            if node.type != "Input":
                valid = np.append(valid, node.id)
        return valid


    def get_valid_inputs(self):
        valid = np.arange(0)
        for node in self.node_list:
            if node.type != "Output":
                valid = np.append(valid, node.id)
        return valid

    def num_nodes_connected_to(self, node_id):
        num_connected = 0
        for c in self.connection_list:
            if c.output_id == node_id:
                num_connected += 1
        return num_connected

    def connections_connected_to(self, node_id):
        cons = self.get_dummy_clist()
        cons.pop(0)
        for c in self.connection_list:
            if c.output_id == node_id:
                cons.append(c)
        return cons

    def num_active_connections(self):
        sum = 0
        for c in self.connection_list:
            if c.enabled:
                sum += 1
        return sum


    def show_net(self):
        for i in range(1, self.next_id):
            n = self.find_node(i)
            if n is not None:
                print(n.to_string())
        print("\n")


    




