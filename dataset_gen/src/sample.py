import importlib.util
import sys

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
            

    def __getitem__(self, idx):
        return self.var_vals