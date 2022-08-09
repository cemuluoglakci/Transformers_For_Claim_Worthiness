

class WandbWrapper():
    def __init__(self, config, constants):
        self.config = config
        self.device = constants.device
        self.parent_dir = constants.parent_dir
