import torch
import json


class UserData:
    def __init__(self, train_location, test_location) -> None:
        self.data = self.get_data(train_location, test_location)

    def get_data(self, train_location, test_location):
        local_train_data = self.load_user_data(train_location)
        local_test_data = self.load_user_data(test_location)

        X_train, y_train, X_test, y_test = local_train_data['0']['x'], \
            local_train_data['0']['y'], local_test_data['0']['x'], local_test_data['0']['y']

        X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.float32)
        X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)

        train_samples, test_samples = len(y_train), len(y_test)

        return {
            'x_train': X_train,
            'y_train': y_train,
            'x_test': X_test,
            'y_test': y_test,
            'train_samples': train_samples,
            'test_samples': test_samples
        }

    def load_user_data(self, local_path):
        local_data = {}
        with open(local_path, 'r') as f:
            data = json.load(f)
            local_data.update(data['user_data'])
        return local_data
