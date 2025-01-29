"""
Client script for CIFAR-10 CNN Federated Learning
"""

from blockchain import *
from uuid import uuid4
import requests
import time
import pickle
import codecs

import cifar_data_extractor as dataext  # 改为你放置 'get_dataset_details','load_data'的脚本
from federatedlearner import NNWorker, reset

class Client:
    def __init__(self, miner, dataset_path):
        self.id = str(uuid4()).replace('-','')
        self.miner = miner
        self.dataset = self.load_dataset(dataset_path)

    def load_dataset(self, path):
        if not path:
            return None
        return dataext.load_data(path)

    def get_chain(self):
        resp = requests.get(f'http://{self.miner}/chain')
        if resp.status_code==200:
            return resp.json()['chain']
        return []

    def get_last_block(self):
        chain = self.get_chain()
        return chain[-1]

    def get_miner_status(self):
        r = requests.get(f'http://{self.miner}/status')
        if r.status_code==200:
            return r.json()
        return {}

    def get_model(self, hblock):
        resp = requests.post(f'http://{self.miner}/model',
                             json={'hblock':hblock})
        if resp.json()['valid']:
            m_data = resp.json()['model']
            return dict(pickle.loads(codecs.decode(m_data.encode(), "base64")))
        print("Invalid model!")
        return None

    def update_model(self, model_params, local_epochs):
        reset()
        t0 = time.time()
        # 注意: self.dataset 里 'train_images' shape=(N,32,32,3), 'train_labels'=(N,10)
        worker = NNWorker(
            self.dataset['train_images'],
            self.dataset['train_labels'],
            self.dataset['test_images'],
            self.dataset['test_labels'],
            len(self.dataset['train_images']),
            self.id,
            steps=local_epochs
        )
        worker.build(model_params)  # CNN build
        worker.train()
        updated_params = worker.get_model()
        acc = worker.evaluate()
        worker.close()
        return updated_params, acc, (time.time() - t0)

    def send_update(self, update, cmp_time, baseindex):
        requests.post(f'http://{self.miner}/transactions/new',
                      json={
                        'client': self.id,
                        'baseindex': baseindex,
                        'update': codecs.encode(pickle.dumps(sorted(update.items())), "base64").decode(),
                        'datasize': len(self.dataset['train_images']),
                        'computing_time': cmp_time
                      })

    def work(self, device_id, global_rounds, local_epochs):
        last_model = -1
        for i in range(global_rounds):
            # 等待矿工进入 receiving 状态
            wait = True
            while wait:
                status = self.get_miner_status()
                if not status:
                    time.sleep(1)
                    continue
                if status['status']!="receiving" or last_model==status['last_model_index']:
                    print("waiting miner receiving...")
                    time.sleep(2)
                else:
                    wait=False

            # 获取最新区块
            hblock = self.get_last_block()
            baseindex = hblock['index']
            print("Global model accuracy", hblock['accuracy'])
            last_model = baseindex

            # 下载模型 & 本地训练
            model = self.get_model(hblock)
            new_params, local_acc, ctime = self.update_model(model, local_epochs)

            # 保存一下
            # with open(f"clients/device{device_id}_model_v{i}.pkl","wb") as f:
            #     pickle.dump(new_params,f)

            print(f"[{device_id}] local update round{i}, local_acc={local_acc:.3f}")
            self.send_update(new_params, ctime, baseindex)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--miner',default='127.0.0.1:5000')
    parser.add_argument('-d','--dataset',default='')
    parser.add_argument('-e','--epoch',default='')
    parser.add_argument('-gr','--global_rounds',default=5,type=int)
    parser.add_argument('-le','--local_epochs',default=1,type=int)
    args = parser.parse_args()

    c = Client(args.miner, args.dataset)
    dataext.get_dataset_details(c.dataset)
    device_id = c.id[:4]
    print("Start working, dev=", device_id)
    c.work(device_id, args.global_rounds, args.local_epochs)
