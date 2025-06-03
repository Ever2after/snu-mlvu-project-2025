import importlib.util
import sys
import collections

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

        # get names of all variables
        what_to_change = getattr(module, "what_to_change", None)
        outvar_names = list(what_to_change.values())

        # import config for the variables
        self.invars = getattr(module, "values", None)

        var_names_from_inver = []
        for i in self.invars.keys():
            var_names_from_inver.extend(i)
        if set(outvar_names) != set(var_names_from_inver):
            # check if names in "values" are equal to names at "what to change"
            raise AssertionError(f"{set(outvar_names)} != {set(var_names_from_inver)}")

        self.lengths = collections.OrderedDict()
        self.total_combs = 1
        for var in self.invars:
            self.lengths[var] = len(self.invars[var])
            self.total_combs *= self.lengths[var]
    
    def __len__(self):
        return self.total_combs
    
    def __getitem__(self, idx):
        sample_res = {}
        for var_set in self.lengths:
            length = self.lengths[var_set]
            var_set_sample = self.invars[var_set][idx % length]
            for i, var in enumerate(var_set):
                sample_res[var] = var_set_sample[i]

            idx //= length
        
        return sample_res