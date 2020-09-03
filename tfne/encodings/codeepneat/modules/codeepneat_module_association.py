from .codeepneat_module_densedropout import CoDeepNEATModuleDenseDropout
from .codeepneat_module_conv2dmaxpool2ddropout import CoDeepNEATModuleConv2DMaxPool2DDropout

# Dict associating the string name of the module when referenced in CoDeepNEAT config with the concrete instance of
# the respective module
MODULES = {
    'DenseDropout': CoDeepNEATModuleDenseDropout,
    'Conv2DMaxPool2DDropout': CoDeepNEATModuleConv2DMaxPool2DDropout
}
