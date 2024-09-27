import json
import lmdb
import pickle
from torch.utils.data import Dataset


class FrameDiff_Dataset(Dataset):
    def __init__(self, cache_dir, is_training = True):
        super().__init__()
        self.is_training = is_training
        self.cache_dir = cache_dir
        self.local_cache = None
        self.num_data = None
        self.cache_to_memory()

    def __len__(self):
        return len(self.num_data)

    def __getitem__(self, idx):
        chain_feats, gt_bb_rigid, pdb_name, csv_row = self.get_cache_csv_row(idx)
        #TODO Load the data item

    def cache_to_memory(self):
        print(f"Loading cache from local dataset @ {self.cache_dir}")
        self.local_cache = lmdb.open(self.cache_dir)
        result_tuples = []
        with self.local_cache.begin() as txn:
            for _, value in txn.cursor():
                result_tuples.append(pickle.loads(value))

        # Split the dataset
        if self.is_training:
            result_tuples = result_tuples[:-100]
        else:
            result_tuples = result_tuples[-100:]


        self.num_data = len(result_tuples)
        def _get_list(idx):
            return list(map(lambda x: x[idx], result_tuples))
        self.chain_ftrs = _get_list(0)
        self.gt_bb_rigid_vals = _get_list(1)
        self.pdb_names = _get_list(2)
        self.csv_rows = _get_list(3)

    def get_cache_csv_row(self, idx):
        return (
            self.chain_ftrs[idx],
            self.gt_bb_rigid_vals[idx],
            self.pdb_names[idx],
            self.csv_rows[idx],
        )




if __name__ == '__main__':
    instance = FrameDiff_Dataset(cache_dir='/home/yu/HENU/westlake/Lit-ProteinDF/preprocess/.cache/jsonl')
