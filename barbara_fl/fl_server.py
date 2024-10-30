import socket
from enum import Enum
import json
import pickle
import struct
import argparse
from threading import Thread, Lock

from socket_methods import receiver
from neural_net_helper import NeuralNet

import torch


class ServerStage(Enum):
    WAITING_FIRST_CLIENT = 1
    WAIT_T_SECONDS = 2
    TRAINING_MODEL = 3


class Server:

    def __init__(self, model: NeuralNet, time_to_wait, num_rounds, host='localhost',
                 port=50052) -> None:
        self.HOST = host
        self.PORT = port
        self.connected_clients: list[socket.socket] = []
        self.clients_info = {}
        self.current_stage: ServerStage = ServerStage.WAITING_FIRST_CLIENT
        self.global_model = model
        self.num_rounds = num_rounds
        self.total_data_size = 0
        self.time_to_wait = time_to_wait
        self.mutex = Lock()

    def handle_connection(self, conn: socket.socket, addr):
        print(f'Node has connected with address: {addr}')

        length_packet = conn.recv(8)
        if not length_packet:
            raise RuntimeError('socket connection broken')

        length_packet = struct.unpack('>Q', length_packet)[0]
        metadata = receiver(conn, length_packet)
        json_data = json.loads(metadata)

        match json_data['header']:
            case "<METADATA>":
                if self.current_stage == ServerStage.TRAINING_MODEL:
                    data = {'header': '<ERROR>',
                            'msg': 'The server has already started the training. You cannot participate now'}
                    json_data = json.dumps(data).encode()
                    lenght_packet = struct.pack('>Q', len(json_data))
                    conn.sendall(lenght_packet)
                    conn.sendall(json_data)
                    conn.close()
                    return

                client_id = json_data['id']
                data_size = json_data['data_size']

                with self.mutex:
                    self.clients_info[client_id] = {
                        'data_size': data_size, 'addr': addr}
                    self.total_data_size += data_size
                    self.connected_clients.append(conn)
                    self.current_stage = ServerStage.WAIT_T_SECONDS

                data = {'header': '<SUCCESS>',
                        'msg': 'You are now participating in the training process'}
                json_data = json.dumps(data).encode()
                length_packet = struct.pack('>Q', len(json_data))
                conn.sendall(length_packet)
                conn.sendall(json_data)

            case _:
                data = {'header': '<ERROR>',
                        'msg': 'Wrong header format. Should start with <METADATA>'}
                json_data = json.dumps(data).encode()
                lenght_packet = struct.pack('>Q', len(json_data))
                conn.sendall(lenght_packet)
                conn.sendall(json_data)
                conn.close()

    def broadcast_global_model(self):
        print('Broadcasting new global Model...')

        for conn in self.connected_clients:
            try:
                data = {'header': '<MODEL>', 'model': self.global_model}
                serialized_data = pickle.dumps(data)
                lenght_packet = struct.pack('>Q', len(serialized_data))

                conn.sendall(lenght_packet)
                conn.sendall(serialized_data)
            except Exception as excp:
                print(f'Connection lost with client: {excp}')

    def close_server(self):
        for conn in self.connected_clients:
            data = {'header': '<FINISHED>'}
            serialized_data = pickle.dumps(data)
            length_packet = struct.pack('>Q', len(serialized_data))

            conn.sendall(length_packet)
            conn.sendall(serialized_data)

    def listen_for_models(self, conn: socket.socket) -> dict[str, NeuralNet]:
        num_models_received = 0
        models_received: dict[str, NeuralNet] = {}
        while num_models_received < len(self.connected_clients):
            conn_model, _ = conn.accept()

            lenght_packet = conn_model.recv(8)
            msg_len = struct.unpack('>Q', lenght_packet)[0]

            data = receiver(conn_model, msg_len)
            deserialized_data = pickle.loads(data)

            client_id = deserialized_data['id']
            models_received[client_id] = deserialized_data['model']

            print(f'Getting local model from client {client_id}')

            num_models_received += 1

        return models_received

    def fedAVG(self, models: dict[str, NeuralNet]):
        print('Aggregating new global model')

        for global_param in self.global_model.parameters():
            global_param.data = torch.zeros_like(global_param.data)

        for id, model in models.items():
            for global_param, user_param in zip(self.global_model.parameters(), model.parameters()):
                global_param.data += user_param.data * \
                    (self.clients_info[id]['data_size'] / self.total_data_size)

    def run(self):
        print('The Federated server is running')

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.HOST, self.PORT))
            s.listen()

            # waiting for first client
            while self.current_stage == ServerStage.WAITING_FIRST_CLIENT:
                conn, addr = s.accept()
                self.handle_connection(conn, addr)

            # waiting T seconds for new clients
            s.settimeout(self.time_to_wait)
            while True:
                try:
                    conn, addr = s.accept()
                    # self.handle_connection(conn, addr)
                    Thread(target=self.handle_connection, args=[conn, addr]).start()
                except socket.timeout:
                    break

            self.current_stage = ServerStage.TRAINING_MODEL

            # starting training
            for num_round in range(1, self.num_rounds + 1):
                print(f'\nGlobal Iteration {num_round}')
                print(f'Total number of clients: {len(self.connected_clients)}')

                self.broadcast_global_model()
                models_received = self.listen_for_models(s)
                self.fedAVG(models_received)

        self.close_server()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--server-port', type=int,
                        help='The port for the FL server', default=50052)
    parser.add_argument('--num-rounds', type=int,
                        help='Number of rounds the server will conduct in the trainning process',
                        default=30)
    parser.add_argument('--time-to-wait', type=int,
                        help='Time in seconds to wait for new clients before the trainning process start',
                        default=10)
    parser.add_argument('--use-cuda', type=bool, default=True,
                        help='Choose if you want to use Cuda during the training proccess. Default to True')

    args = parser.parse_args()
    port = args.server_port
    time = args.time_to_wait
    rounds = args.num_rounds

    device_opt = 'gpu' if args.use_cuda else 'cpu'

    match device_opt:
        case 'gpu':
            device = torch.device('cuda')
        case _:
            device = torch.device('cpu')

    global_model = NeuralNet().to(device)

    server = Server(time_to_wait=time, num_rounds=rounds,
                    port=port, model=global_model)
    server.run()
