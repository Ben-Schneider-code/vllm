from ..model.abc import modelling_abc
def model_factory(model_args, data_args, training_args):

    if training_args.loss_type == "last_token":
        return modelling_abc.IVLLT
    elif training_args.loss_type == "mean_token":
        return modelling_abc.IVLMT
    else: raise Exception("NotImplementedError")
    