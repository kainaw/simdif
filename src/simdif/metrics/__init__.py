# simdif/metrics/__init__.py
import pkgutil
import importlib

# Look at every .py file in this folder
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    # Import the module (e.g., .jaccard)
    module = importlib.import_module(f".{module_name}", package=__name__)

    # Take everything inside that module and add it to THIS namespace
    globals().update({
        name: getattr(module, name)
        for name in dir(module)
        if not name.startswith('_')
    })
