"""
 - Blockchain for Federated Learning -
   Blockchain script 
"""

import hashlib
import json
import time
from urllib.parse import urlparse
import requests
import random
import codecs
import federated_data_extractor as dataext
from federatedlearner import *
from miner import app

accuracy_list = []
'''
该函数根据客户端发送的模型更新计算全局模型。每个客户端发送的更新会按比例调整全局模型的参数，然后通过测试数据集评估全局模型的准确性。
updates 包含各客户端的模型更新。函数根据学习率 (lrate) 和客户端数量按比例聚合这些更新。
最终返回全局模型的准确性和新的模型参数。
'''


# todo 直接进行参数聚合更新操作，可修改为基于准确率/信誉积分/训练数据量等
# lrate 1

# todo 测试非联邦学习 直接以第一个局部模型作为结果（此时不分割数据集），相当于不使用聚合
def compute_upd_1(weights, base, updates, lrate):
    # 计算聚合更新操作
    upd = dict()

    for client in updates.keys():
        for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
            model = updates[client].update
            # 计算参数聚合结果
            upd[x] = model[x]
    upd["size"] = 0
    return upd


# FedAvg算法 weight为大家平均（假设数据量相同）, 结果同正常一致
def compute_upd_2(weights, base, updates, lrate):
    # weight全部相同
    for client in weights.keys():
        weights[client] = 1 / len(weights)

    # 计算聚合更新操作
    upd = dict()
    for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
        upd[x] = np.array(base[x], copy=False)
        upd[x].fill(0)

    # number_of_clients = len(updates)  # 局部模型数目

    for client in updates.keys():
        for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
            model = updates[client].update
            # 计算参数聚合结果
            upd[x] += (lrate * weights[client]) * model[x]

    upd["size"] = 0
    return upd


# 基于准确率进行聚合 || 基于准确率和准确率变化值进行聚合（计算weight）
def compute_upd_3(weights, base, updates, lrate):
    lrate = 1
    # 计算聚合更新操作
    upd = dict()
    for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
        upd[x] = np.array(base[x], copy=False)
        upd[x].fill(0)

    for client in updates.keys():
        for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
            model = updates[client].update
            # 计算参数聚合结果
            upd[x] += (lrate * weights[client]) * (model[x])

    upd["size"] = 0
    return upd

# 基于准确率进行聚合 || 基于准确率和准确率变化值进行聚合（外层负责计算）
def compute_upd_3(weights, base, updates, lrate):
    lrate = 1
    # 计算聚合更新操作
    upd = dict()
    for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
        upd[x] = np.array(base[x], copy=False)
        upd[x].fill(0)

    for client in updates.keys():
        for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
            model = updates[client].update
            # 计算参数聚合结果
            upd[x] += (lrate * weights[client]) * (model[x])

    upd["size"] = 0
    return upd


# 废弃: 在FedAvg基础之上，添加了上一轮全局模型的影响（这里不对，参数会变为原来二倍，导致参数无法收敛
def compute_upd_n1(weights, base, updates, lrate):
    lrate = 1
    # 计算聚合更新操作
    upd = dict()
    for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
        upd[x] = np.array(base[x], copy=False)
        upd[x].fill(0)

    # number_of_clients = len(updates)  # 局部模型数目

    for client in updates.keys():
        for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
            model = updates[client].update
            # 计算参数聚合结果
            upd[x] += (lrate * weights[client]) * (model[x] + base[x])

    upd["size"] = 0
    return upd


def compute_global_model(base_block, updates, lrate):
    base = base_block.basemodel
    accuracy_local_all = dict()
    # 1. 每个局部模型在训练集上计算准确率
    for client, update in updates.items():
        # 在测试集上计算模型准确率
        reset()
        dataset = dataext.load_data("data/mnist.d")
        worker = NNWorker(None,
                          None,
                          dataset['test_images'],
                          dataset['test_labels'],
                          0,
                          "validation")
        worker.build(update.update)
        evaluateAccuracy = worker.evaluate()
        worker.close()
        accuracy_local_all[client] = evaluateAccuracy

    # 计算每个模型的权重，todo 可以把数据分割，在一个节点上模拟多个节点在不同验证集下准确率变化的结果
    weights = {}

    total_weight = 0

    # KR1: 基于准确率和准确率变化计算结果
    alpha = 0.1
    beta = 0.9
    for client, update in updates.items():
        # accuracy_local_all[client] 是局部模型的准确率, base_block是上一轮全局模型准确率
        acc_i_t = accuracy_local_all[client]
        acc_change_i_t = acc_i_t - base_block.accuracy  # 计算准确率变化值
        weight = alpha * acc_i_t + beta * acc_change_i_t
        app.logger.info(
            "client:{}, local.accuracy: {}, base.accuracy:{}, weight: {}".format(client, accuracy_local_all[client],
                                                                                 base_block.accuracy, weight))
        weights[client] = weight
        total_weight += weight

    # 计算权重占比
    for client in updates.keys():
        weights[client] = weights[client] / total_weight
        app.logger.info("client: {}, weights[client]: {}".format(client, weights[client]))

    # todo 在这个位置更改计算方式
    upd = compute_upd_2(weights, base, updates, lrate)

    tf.reset_default_graph()
    dataset = dataext.load_data("data/mnist.d")
    worker = NNWorker(None,
                      None,
                      dataset['test_images'],
                      dataset['test_labels'],
                      0,
                      "validation")
    worker.build(upd)
    accuracy = worker.evaluate()
    worker.close()
    return accuracy, upd


def find_len(text, strk):
    '''
    Function to find the specified string in the text and return its starting position 
    as well as length/last_index
    '''
    return text.find(strk), len(strk)


'''
    解释:
    Update 类记录了来自某个客户端的模型更新。每个更新包括客户端 ID (client)、模型更新的基本索引 (baseindex)、实际的模型更新 (update)、数据量和计算时间。
    from_string 方法允许从存储的字符串还原出 Update 对象，这样可以轻松地在网络上传输更新。
    __str__ 方法将 Update 对象转换为字符串，以便存储或传输。
'''


class Update:
    def __init__(self, client, baseindex, update, datasize, computing_time, timestamp=time.time()):
        '''
        Function to initialize the update string parameters
        '''
        self.timestamp = timestamp
        self.baseindex = baseindex
        self.update = update
        self.client = client
        self.datasize = datasize
        self.computing_time = computing_time

    @staticmethod
    def from_string(metadata):
        '''
        Function to get the update string values
        '''
        i, l = find_len(metadata, "'timestamp':")
        i2, l2 = find_len(metadata, "'baseindex':")
        i3, l3 = find_len(metadata, "'update': ")
        i4, l4 = find_len(metadata, "'client':")
        i5, l5 = find_len(metadata, "'datasize':")
        i6, l6 = find_len(metadata, "'computing_time':")
        baseindex = int(metadata[i2 + l2:i3].replace(",", '').replace(" ", ""))
        update = dict(pickle.loads(codecs.decode(metadata[i3 + l3:i4 - 1].encode(), "base64")))
        timestamp = float(metadata[i + l:i2].replace(",", '').replace(" ", ""))
        client = metadata[i4 + l4:i5].replace(",", '').replace(" ", "")
        datasize = int(metadata[i5 + l5:i6].replace(",", '').replace(" ", ""))
        computing_time = float(metadata[i6 + l6:].replace(",", '').replace(" ", ""))
        return Update(client, baseindex, update, datasize, computing_time, timestamp)

    def __str__(self):
        '''
        Function to return the update string values in the required format
        '''
        return "'timestamp': {timestamp},\
            'baseindex': {baseindex},\
            'update': {update},\
            'client': {client},\
            'datasize': {datasize},\
            'computing_time': {computing_time}".format(
            timestamp=self.timestamp,
            baseindex=self.baseindex,
            update=codecs.encode(pickle.dumps(sorted(self.update.items())), "base64").decode(),
            client=self.client,
            datasize=self.datasize,
            computing_time=self.computing_time
        )


'''
    解释:
    Block 类表示区块链中的一个区块，它存储着某一轮联邦学习的全局模型、更新信息和其他元数据（如区块的挖矿者）。
    from_string 方法允许将区块从字符串反序列化，还原出一个 Block 对象。
    __str__ 方法将区块序列化为字符串，以便存储到本地文件系统或者传输给其他节点。
'''


class Block:
    def __init__(self, miner, index, basemodel, accuracy, updates, timestamp=time.time()):
        '''
        Function to initialize the update string parameters per created block
        '''
        self.index = index
        self.miner = miner
        self.timestamp = timestamp
        self.basemodel = basemodel
        self.accuracy = accuracy
        self.updates = updates

    @staticmethod
    def from_string(metadata):
        '''
        Function to get the update string values per block
        '''
        i, l = find_len(metadata, "'timestamp':")
        i2, l2 = find_len(metadata, "'basemodel': ")
        i3, l3 = find_len(metadata, "'index':")
        i4, l4 = find_len(metadata, "'miner':")
        i5, l5 = find_len(metadata, "'accuracy':")
        i6, l6 = find_len(metadata, "'updates':")
        i9, l9 = find_len(metadata, "'updates_size':")
        index = int(metadata[i3 + l3:i4].replace(",", '').replace(" ", ""))
        miner = metadata[i4 + l4:i].replace(",", '').replace(" ", "")
        timestamp = float(metadata[i + l:i2].replace(",", '').replace(" ", ""))
        basemodel = dict(pickle.loads(codecs.decode(metadata[i2 + l2:i5 - 1].encode(), "base64")))
        accuracy = float(metadata[i5 + l5:i6].replace(",", '').replace(" ", ""))
        su = metadata[i6 + l6:i9]
        su = su[:su.rfind("]") + 1]
        updates = dict()
        for x in json.loads(su):
            isep, lsep = find_len(x, "@|!|@")
        updates[x[:isep]] = Update.from_string(x[isep + lsep:])
        updates_size = int(metadata[i9 + l9:].replace(",", '').replace(" ", ""))
        return Block(miner, index, basemodel, accuracy, updates, timestamp)

    def __str__(self):
        '''
        Function to return the update string values in the required format per block
        '''
        return "'index': {index},\
            'miner': {miner},\
            'timestamp': {timestamp},\
            'basemodel': {basemodel},\
            'accuracy': {accuracy},\
            'updates': {updates},\
            'updates_size': {updates_size}".format(
            index=self.index,
            miner=self.miner,
            basemodel=codecs.encode(pickle.dumps(sorted(self.basemodel.items())), "base64").decode(),
            accuracy=self.accuracy,
            timestamp=self.timestamp,
            updates=str([str(x[0]) + "@|!|@" + str(x[1]) for x in sorted(self.updates.items())]),
            updates_size=str(len(self.updates))
        )


'''
    解释:
    Blockchain 类是核心逻辑，负责维护整个区块链。每个节点都拥有一条区块链，其中包含全局模型和每轮训练的更新。
    make_block 方法用于生成新块，在收到足够的客户端更新后，区块链会聚合这些更新，并生成新的全局模型。
'''


class Blockchain(object):
    def __init__(self, miner_id, base_model=None, gen=False, update_limit=10, time_limit=1800):
        super(Blockchain, self).__init__()
        self.miner_id = miner_id
        self.curblock = None
        self.hashchain = []
        self.current_updates = dict()
        self.update_limit = update_limit
        self.time_limit = time_limit

        if gen:
            genesis, hgenesis = self.make_block(base_model=base_model, previous_hash=1)
            self.store_block(genesis, hgenesis)
        self.nodes = set()

    def register_node(self, address):
        if address[:4] != "http":
            address = "http://" + address
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)
        app.logger.info("Registered node", address)

    def make_block(self, previous_hash=None, base_model=None):
        accuracy = 0
        basemodel = None
        time_limit = self.time_limit
        update_limit = self.update_limit
        if len(self.hashchain) > 0:
            update_limit = self.last_block['update_limit']
            time_limit = self.last_block['time_limit']
        if previous_hash == None:
            previous_hash = self.hash(str(sorted(self.last_block.items())))
        if base_model != None:
            accuracy = base_model['accuracy']
            basemodel = base_model['model']
        elif len(self.current_updates) > 0:
            base_block = self.curblock
            accuracy, basemodel = compute_global_model(base_block, self.current_updates, 1)
        index = len(self.hashchain) + 1
        block = Block(
            miner=self.miner_id,
            index=index,
            basemodel=basemodel,
            accuracy=accuracy,
            updates=self.current_updates
        )
        app.logger.info("index{}, global model accuracy{}".format(index, accuracy))

        # 打印全局模型准确率
        global accuracy_list
        accuracy_list.append("index{}, global model accuracy{}".format(index, accuracy))
        app.logger.info("\n".join(accuracy_list))

        hashblock = {
            'index': index,
            'hash': self.hash(str(block)),
            'proof': random.randint(0, 100000000),
            'previous_hash': previous_hash,
            'miner': self.miner_id,
            'accuracy': str(accuracy),
            'timestamp': time.time(),
            'time_limit': time_limit,
            'update_limit': update_limit,
            'model_hash': self.hash(codecs.encode(pickle.dumps(sorted(block.basemodel.items())), "base64").decode())
        }
        return block, hashblock

    def store_block(self, block, hashblock):
        if self.curblock:
            with open("blocks/federated_model" + str(self.curblock.index) + ".block", "wb") as f:
                pickle.dump(self.curblock, f)
        self.curblock = block
        self.hashchain.append(hashblock)
        self.current_updates = dict()
        return hashblock

    def new_update(self, client, baseindex, update, datasize, computing_time):
        self.current_updates[client] = Update(
            client=client,
            baseindex=baseindex,
            update=update,
            datasize=datasize,
            computing_time=computing_time
        )
        return self.last_block['index'] + 1

    @staticmethod
    def hash(text):
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def last_block(self):
        return self.hashchain[-1]

    # pow工作量证明
    # todo 可修改为别的证明方式
    def proof_of_work(self, stop_event):
        block, hblock = self.make_block()
        stopped = False
        while self.valid_proof(str(sorted(hblock.items()))) is False:
            if stop_event.is_set():
                stopped = True
                break
            hblock['proof'] += 1
            if hblock['proof'] % 1000 == 0:
                # print("mining",hblock['proof']) # 暂时不打印挖矿过程
                pass
        if stopped == False:
            self.store_block(block, hblock)
        if stopped:
            print("Stopped")
        else:
            pass
        return hblock, stopped

    @staticmethod
    def valid_proof(block_data):
        guess_hash = hashlib.sha256(block_data.encode()).hexdigest()
        k = "00"  # 调整挖矿难度到比较快的水平，原来是100000次
        return guess_hash[:len(k)] == k

    def valid_chain(self, hchain):
        last_block = hchain[0]
        curren_index = 1
        while curren_index < len(hchain):
            hblock = hchain[curren_index]
            if hblock['previous_hash'] != self.hash(str(sorted(last_block.items()))):
                print("prev_hash diverso", curren_index)
                return False
            if not self.valid_proof(str(sorted(hblock.items()))):
                print("invalid proof", curren_index)
                return False
            last_block = hblock
            curren_index += 1
        return True

    # resolve_conflicts 方法实现了区块链的共识机制。当多个节点之间发生分叉时，节点会选择最长且有效的链来替换自己的区块链。这确保了整个系统中的一致性。
    def resolve_conflicts(self, stop_event):
        neighbours = self.nodes
        new_chain = None
        bnode = None
        max_length = len(self.hashchain)
        for node in neighbours:
            response = requests.get('http://{node}/chain'.format(node=node))
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
                    bnode = node
        if new_chain:
            stop_event.set()
            self.hashchain = new_chain
            hblock = self.hashchain[-1]
            resp = requests.post('http://{node}/block'.format(node=bnode),
                                 json={'hblock': hblock})
            self.current_updates = dict()
            if resp.status_code == 200:
                if resp.json()['valid']:
                    self.curblock = Block.from_string(resp.json()['block'])
            return True
        return False
