import pytest
import sys
# Manually run the pytest main function
print("Running pytest from pytest_corenrn_runner.py ", sys.argv[2:])
sys.exit(pytest.main(sys.argv[2:]))