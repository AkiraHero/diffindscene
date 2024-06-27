import os
import logging
import traceback

path = os.path.dirname(os.path.abspath(__file__))
py_list = []
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if name.endswith(".py") and not name.endswith("__init__.py"):
            rel_dir = os.path.relpath(root, path)
            if rel_dir != ".":
                rel_file = os.path.join(rel_dir, name)
            else:
                rel_file = name
            py_list.append(rel_file)
for py in py_list:
    mod_name = ".".join([__name__, *(py.split("/"))])
    mod_name = mod_name[:-3]
    try:
        mod = __import__(mod_name, fromlist=[mod_name])
    except ModuleNotFoundError as e:
        logging.error(traceback.format_exc())
        logging.debug("Fail to import submodule:".format(mod_name))
        continue
    classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
    for cls in classes:
        if "model" in str(cls):
            globals()[cls.__name__] = cls
