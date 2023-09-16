import ivy
from ivy.functional.frontends.numpy.func_wrapper import inputs_to_ivy_arrays

# reapeated utility function
@inputs_to_ivy_arrays
def type_of_target(y, input_name="y"):
    # purely utility function
    y = ivy.array(y)
    unique_vals = len(ivy.unique_values(y))
    if y.ndim == 2 and y.shape[1] > 1 and unique_vals <= 2:
        return "multilabel-indicator"
    if y.ndim not in (1, 2):
        return "unknown"
    if ivy.is_float_dtype(y) and ivy.any(ivy.not_equal(y, y.astype("int64"))):
        return "continuous"
    else:
        if unique_vals > 2:
            return "multiclass"
        else:
            return "binary"
