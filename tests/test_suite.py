"""Collects the shared fnn-testkit suite into this repository's test run.

Every backend is validated by the *same* tests; see the testkit README for the
list.  Options: --device, --dtype, --blas, --option KEY=VALUE, --run-slow.
"""
from fnn_testkit.tests.test_activations import *  # noqa: F401,F403
from fnn_testkit.tests.test_devices import *  # noqa: F401,F403
from fnn_testkit.tests.test_forward import *  # noqa: F401,F403
from fnn_testkit.tests.test_gradients import *  # noqa: F401,F403
from fnn_testkit.tests.test_integration import *  # noqa: F401,F403
from fnn_testkit.tests.test_optimizers import *  # noqa: F401,F403
from fnn_testkit.tests.test_parity import *  # noqa: F401,F403
from fnn_testkit.tests.test_profile import *  # noqa: F401,F403
from fnn_testkit.tests.test_regularization import *  # noqa: F401,F403
from fnn_testkit.tests.test_training import *  # noqa: F401,F403

# the star imports also bring in module-level `pytestmark`s (compiled_only) that
# must not apply to the whole file
pytestmark = []
