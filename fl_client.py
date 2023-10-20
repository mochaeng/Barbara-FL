import socket
import json
import struct
import pickle
import argparse

import client_helper
from socket_methods import receiver
from neural_net_helper import NetHelper


class Client:
    def __init__(self, id: str, user_data: client_helper.UserData, device,
                 host='localhost', server_port=50052) -> None:

        self.SERVER_HOST = host
        self.SERVER_PORT = server_port
        self.CLIENT_HOST = host
        self.user_data = user_data
        self.id: str = id
        self.device = device

    def sending_initial_metadata(self, conn: socket.socket):
        data_size = self.user_data.data['train_samples']
        metadata = {
            'header': '<METADATA>', 'id': self.id, 'data_size': data_size
        }

        json_data = json.dumps(metadata).encode()
        msg_len = struct.pack('>Q', len(json_data))
        conn.sendall(msg_len)
        conn.sendall(json_data)

    def prepare_model_to_send(self, net_helper: NetHelper) -> bytes:
        data = {
            'header': '<MODEL>', 'id': self.id, 'model': net_helper.model
        }
        serialized_data = pickle.dumps(data)

        return serialized_data

    def start(self):

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.connect((self.SERVER_HOST, self.SERVER_PORT))

            self.sending_initial_metadata(s)

            length_packet = s.recv(8)
            msg_len = struct.unpack('>Q', length_packet)[0]
            data = receiver(s, msg_len)
            json_data = json.loads(data)

            match json_data['header']:
                case '<SUCCESS>':
                    print(f"message sent by the server: {json_data['msg']}")
                case _:
                    print(f"message sent by the server: {json_data['msg']}")
                    return

            # training
            while True:
                length_packet = s.recv(8)
                msg_len = struct.unpack('>Q', length_packet)[0]

                data = receiver(s, msg_len)
                deserialized_data = pickle.loads(data)

                if deserialized_data['header'] == '<FINISHED>':
                    return

                print(f'\nI am client {self.id}')
                print('Receiving new global model')

                global_model = deserialized_data['model']

                net_helper = NetHelper(global_model, self.user_data.data, self.device)
                training_loss = net_helper.train_loss()
                test_loss = net_helper.test()

                print(f'Training loss: {training_loss:.2f}')
                print(f'Testing accuracy: {test_loss:.2f}')
                print('Local training...')

                net_helper.train()

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as model_socket:
                    model_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    model_socket.connect((self.SERVER_HOST, self.SERVER_PORT))

                    model_to_send = self.prepare_model_to_send(net_helper)
                    length_packet = struct.pack('>Q', len(model_to_send))

                    model_socket.sendall(length_packet)
                    model_socket.sendall(model_to_send)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--server-port', type=int,
                        help='The port for the FL server', default=50052)
    parser.add_argument('--client-id', type=int,
                        help='The ID of your client. Must be between 1 and 5')
    parser.add_argument('--use-cuda', type=bool, default=True,
                        help='Choose if you want to use Cuda during the training proccess. Default to True')

    args = parser.parse_args()

    if not args.client_id:
        parser.error('You must specify a client ID between 1 and 5')

    client_id = args.client_id
    if client_id > 5 or client_id < 1:
        parser.error('Client ID must be between 1 and 5')

    server_port = args.server_port
    device = 'gpu' if args.use_cuda else 'cpu'

    train_location_path = f'./FLdata/train/mnist_train_client{str(client_id)}.json'
    test_location_path = f'./FLdata/test/mnist_test_client{str(client_id)}.json'

    print(f'Training file: {train_location_path}')

    user_data = client_helper.UserData(train_location_path, test_location_path)

    client = Client(client_id, user_data, device=device, server_port=server_port)
    client.start()
