from robomimic.algo import register_algo_factory_func, WeighingAlgo

class RNNDiscriminator(WeighingAlgo):
    def _create_networks(self):
        pass
    def process_batch_fortraining(self, batch):
        pass
    def train_on_batch(self, batch, epoch, validate=False):
        pass
    def _forward_training(self, batch):
        pass
    def _compute_losses(self, predictions, gt):
        pass
    def _train_step(self, losses):
        pass

