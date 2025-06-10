import importlib.util
import sys
import collections
import pickle
import os
import hashlib
from copy import deepcopy

def make_float(obj):
    obj = deepcopy(obj)
    if isinstance(obj, str) or isinstance(obj, float):
        return obj
    if isinstance(obj, int):
        return float(obj)
    if isinstance(obj, tuple):
        obj = list(obj)
        for i in range(len(obj)):
            obj[i] = make_float(obj[i])
        return tuple(obj)
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = make_float(obj[i])
        return obj
    if isinstance(obj, dict):
        for i in obj:
            i_float = make_float(i)
            obj[i_float] = make_float(obj[i])
        return obj


class SimpleSample:
    def __init__(self, sample_conf_path):
        module_name = "loaded_sample_conf"

        spec = importlib.util.spec_from_file_location(module_name, sample_conf_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # get names of all variables
        what_to_change = getattr(module, "what_to_change", None)
        var_names = what_to_change.values()

        # import config for the variables
        self.var_vals = {}
        for var in var_names:
            set_value = getattr(module, var, None)
            self.var_vals[var] = set_value
    
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.var_vals


class GridSample:
    def __init__(self, sample_conf_path):
        module_name = "loaded_sample_conf"

        spec = importlib.util.spec_from_file_location(module_name, sample_conf_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # get names of all variables
        what_to_change = getattr(module, "what_to_change", None)
        var_names = what_to_change.values()

        # import config for the variables
        self.var_vals = {}
        for var in var_names:
            set_value = getattr(module, var, None)
            self.var_vals[var] = set_value
        
        self.lengths = collections.OrderedDict()
        self.total_combs = 1
        for var in self.var_vals:
            self.lengths[var] = len(self.var_vals[var])
            self.total_combs *= self.lengths[var]
    
    def __len__(self):
        return self.total_combs
    
    def __getitem__(self, idx):
        sample_res = {}
        for var_name in self.lengths:
            length = self.lengths[var_name]
            sample_res[var_name] = self.var_vals[var_name][idx % length]
            idx //= length

        return sample_res


class GridSample_joint:
    def __init__(self, sample_conf_path):
        module_name = "loaded_sample_conf"

        spec = importlib.util.spec_from_file_location(module_name, sample_conf_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # get names of all required variables
        what_to_change = getattr(module, "what_to_change", None)
        outvar_names = list(what_to_change.values())

        # import config for provided variables
        self.invars = getattr(module, "values", None)
        var_names_from_inver = []
        for i in self.invars.keys():
            var_names_from_inver.extend(i)
        
        # check if names in "values" are equal to names at "what to change"
        if set(outvar_names) != set(var_names_from_inver):
            raise AssertionError(f"{set(outvar_names)} != {set(var_names_from_inver)}")
        
        # get "cache variables"
        meshes = []
        with open(os.path.join(os.path.dirname(sample_conf_path), "out_summary.txt")) as f:
            lines = f.readlines()
            is_mesh_region = False
            for line in lines:
                line = line.strip("\n")
                if len(line) < 2: continue

                if not is_mesh_region and line == "MESH":
                    is_mesh_region = True
                elif is_mesh_region and line[0] == "\t" and line[1] != "\t":
                    meshes.append(line[1:])
                elif is_mesh_region and line[0] != "\t":
                    break

        self.var_cache = []
        for var_path in what_to_change:
            first_word = var_path.split(".")[0]
            if first_word in meshes:
                self.var_cache.append(what_to_change[var_path])

        # get raw length
        self.lengths = collections.OrderedDict()
        self.total_combs = 1
        for var in self.invars:
            self.lengths[var] = len(self.invars[var])
            self.total_combs *= self.lengths[var]
    
    def get_hash(self):
        # order the dict
        invars_float = make_float(self.invars)
        keys_in = sorted(list(invars_float.keys()))
        vals_in = [invars_float[var_set] for var_set in keys_in]

        hash_obj = hashlib.sha1(pickle.dumps(keys_in) + pickle.dumps(vals_in))
        return hash_obj.hexdigest()
    
    def get_cache_folder_name(self, idx):
        index_samples = collections.OrderedDict()
        for var_set in self.lengths:
            length = self.lengths[var_set]
            index_samples[var_set] = idx % length
            idx //= length
        
        # get idx for each set that includes cache var
        # OrderedDict should provide deterministic ordering
        res = str(self.get_hash())
        res += "_"
        for var_set in index_samples:
            if not set(var_set).isdisjoint(self.var_cache):
                res += str(index_samples[var_set])
            else:
                res += "n"
            res += "_"
        
        return res[:-1]
    
    def __len__(self):
        return self.total_combs
    
    def __getitem__(self, idx):
        sample_res = {}
        for var_set in self.lengths:
            length = self.lengths[var_set]
            var_set_sample = self.invars[var_set][idx % length]

            # append for each individual var
            for i, var in enumerate(var_set):
                sample_res[var] = var_set_sample[i]

            idx //= length
        
        return sample_res