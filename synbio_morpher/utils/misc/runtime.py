
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
import sys
import gc
import logging
# import psutil 


def clear_caches():
    """ Jax garbage collection https://github.com/google/jax/issues/10828 """
    jax_mods = [module_name for module_name in sys.modules.keys() if module_name.startswith("jax") and module_name not in ["jax.interpreters.partial_eval"]]
    for module_name in jax_mods:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if hasattr(obj, "cache_clear"):
                try:
                    obj.cache_clear()
                except:
                    pass
    gc.collect()


# def clear_caches():
#     process = psutil.Process()
#     if process.memory_info().vms > 4 * 2**30:  # >4GB memory usage
#         for module_name, module in sys.modules.items():
#             if module_name.startswith("jax"):
#                 for obj_name in dir(module):
#                     obj = getattr(module, obj_name)
#                     if hasattr(obj, "cache_clear"):
#                         obj.cache_clear()
#         gc.collect()
