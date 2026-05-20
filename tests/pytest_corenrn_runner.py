# -*- coding: utf-8 -*-
#
# pytest_corenrn_runner.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Launch this script as
```
python_with_multichannel_test pytest_corenrn_runner.py
```
to test coreneuron.
"""

import pytest
import sys

args = [
    "-s",  # Disable capturing (shortcut for --capture=no)
    "-o",
    "norecursedirs=*",  # Override: don't recurse into any directories
    "-o",
    "log_cli=true",  # Override: enable CLI logging
    "-o",
    "log_cli_level=DEBUG",  # Override: set log level
    "test_neurontree.py",  # target tests
    "test_compartmentfitter.py",  # target tests
    "test_compartmenttree.py",  # target tests
    "test_concmechs.py",  # target tests
]

# Manually run the pytest main function
print(f"Running pytest from {sys.argv[2]}")
exit_code = pytest.main(args)  # ""])
sys.exit(exit_code)
