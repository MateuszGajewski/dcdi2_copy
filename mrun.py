import glob
import os
from argparse import Namespace

import mrunner
from mrunner.helpers.client_helper import get_configuration

from dcdi.main import main

mrunner.settings.NEPTUNE_USE_NEW_API = True

if __name__ == "__main__":
    params = get_configuration(
        print_diagnostics=True, with_neptune=True, inject_parameters_to_gin=True
    )
    params.pop("experiment_id")
    namespace = Namespace(**params)
    # namespace.graph_files = prepare_graph_files(namespace.graph_files)
    main(namespace)
