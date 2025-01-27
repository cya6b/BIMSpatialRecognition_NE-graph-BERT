'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2024 Yian Chen <cya187508866962021@163.com>
# License: TBD

from code.base_class.setting import setting


class Settings(setting):
    fold = None
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        # run learning methods
        self.method.data = loaded_data
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        # evaluate learning results
        if self.evaluate is not None:
            self.evaluate.data = learned_result
            self.evaluate.evaluate()

        return None

