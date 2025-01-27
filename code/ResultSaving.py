'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2024 Yian Chen <cya187508866962021@163.com>

from code.base_class.result import result
import pickle


class ResultSaving(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def save(self):
        f = open(self.result_destination_folder_path + self.result_destination_file_name, 'wb')
        pickle.dump(self.data, f)
        f.close()

    def load(self):
        f = open(self.result_destination_folder_path + self.result_destination_file_name, 'rb')
        result = pickle.load(f)
        f.close()
        return result
