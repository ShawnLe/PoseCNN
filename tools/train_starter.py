import _init_paths
from datasets.factory import get_imdb
from fcn.train import get_training_roidb
from gt_data_layer.layer import GtDataLayer

class args_setter(object):
  
  def __init__(self):
    self.imdb_name = 'lov_train'
           
class train_starter(object):

    def __init__(self):

        args = args_setter()
        imdb = get_imdb(args.imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)

        roidb = get_training_roidb(imdb)

        data_layer = GtDataLayer(roidb, imdb.num_classes)
