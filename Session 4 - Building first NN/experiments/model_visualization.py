
from torchviz import make_dot
make_dot(y_predicted.mean(),params=dict(regression_model.named_parameters()))

make_dot(y_predicted.mean(),params=dict(regression_model.named_parameters()),show_attrs=True,show_saved=True)
