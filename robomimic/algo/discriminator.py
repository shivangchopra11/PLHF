## Refer to https://github.com/MaxDu17/BehaviorRetrieval/blob/8088fe9bb6b49f2147b649a440a899624901d789/robomimic/docs/modules/algorithms.md

from robomimic.algo import register_algo_factory_func, WeighingAlgo

@register_algo_factory_func("rnn_discriminator")
def algo_config_to_class(algo_config = None):
    """
    Yields the class for the weighing algorithm. Can be expanded to accomodate more fancier classifiers
    """
    return TemporalEmbeddingWeighter, {}

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

