import omegaconf
import copy
def log_params(experiment, cfg, prefix=None):
    as_dict = copy.deepcopy(cfg.__dict__["_content"])
    del as_dict["_name"]
    inner_dict = {}
    for k, v in as_dict.items():
        if type(v) == omegaconf.dictconfig.DictConfig:
            v_as_dict = copy.deepcopy(v.__dict__["_content"])
            del v_as_dict["_name"]
            experiment.log_parameters(v_as_dict, prefix=prefix)
        elif type(v) == omegaconf.nodes.AnyNode and v._val is not None:
            for key, val in vars(v._val).items():
                inner_dict[key] = val
            del inner_dict["_name"]
        else:
            if prefix is not None:
                k_with_prefix = f"{prefix}_{k}"
                experiment.log_parameter(k_with_prefix, v)
            else:
                experiment.log_parameter(k, v)
    experiment.log_parameters(inner_dict)