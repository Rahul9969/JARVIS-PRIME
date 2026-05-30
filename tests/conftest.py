import pytest

class PytestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name, success, detail=""):
        # Let pytest handle the failure naturally via assert
        assert success, f"{name} failed: {detail}"
        self.passed += 1

    def summary(self):
        pass

@pytest.fixture
def results():
    return PytestResults()
