"""Exercise the whole app flow without needing a Streamlit server in CI."""

from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.sidebar = self
        self.reported_metrics = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cache_data(self, **_):
        return lambda fn: fn

    def __getattr__(self, name):
        if name in {
            "set_page_config", "title", "caption", "info", "header", "subheader",
            "markdown", "warning", "error", "dataframe", "download_button",
            "link_button",
        }:
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def multiselect(self, label, options, default=None, **_):
        return list(default) if default is not None else list(options)

    def toggle(self, label, value=False, **_):
        return value

    def slider(self, label, minimum, maximum, value, step, **_):
        return value

    def columns(self, count):
        return [self] * count

    def metric(self, label, value, **_):
        self.reported_metrics.append((label, value))

    def selectbox(self, label, options, **_):
        return list(options)[0]

    def expander(self, label, **_):
        return self

    def stop(self):
        raise AssertionError("The sample dataset unexpectedly prevented app startup")


class AppSmokeTests(unittest.TestCase):
    def test_all_interface_sections_execute_on_real_snapshot(self):
        fake = FakeStreamlit()
        with patch.dict(sys.modules, {"streamlit": fake}):
            namespace = runpy.run_path(str(PROJECT / "app.py"))
        final = namespace["p"]
        self.assertEqual(len(final), 146)
        self.assertEqual(len(namespace["df"]), 949)
        self.assertTrue(final["short_warning_score"].between(0, 100).all())
        self.assertGreater(len(fake.reported_metrics), 4)


if __name__ == "__main__":
    unittest.main()
