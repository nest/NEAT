# -*- coding: utf-8 -*-

import importlib.util
import json
import shutil
import sys
import types
import uuid
from pathlib import Path

import pytest


def _load_neuronmodel(monkeypatch):
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "neat"
        / "simulations"
        / "neuron"
        / "neuronmodel.py"
    )
    module_name = "neat.simulations.neuron.neuronmodel_under_test"

    neat_pkg = types.ModuleType("neat")
    neat_pkg.__path__ = [str(module_path.parents[2])]

    simulations_pkg = types.ModuleType("neat.simulations")
    simulations_pkg.__path__ = [str(module_path.parents[1])]

    neuron_pkg = types.ModuleType("neat.simulations.neuron")
    neuron_pkg.__path__ = [str(module_path.parent)]

    trees_pkg = types.ModuleType("neat.trees")
    trees_pkg.__path__ = [str(module_path.parents[2] / "trees")]

    morphtree_mod = types.ModuleType("neat.trees.morphtree")
    morphtree_mod.MorphLoc = object

    phystree_mod = types.ModuleType("neat.trees.phystree")
    phystree_mod.PhysTree = object
    phystree_mod.PhysNode = object

    compartmenttree_mod = types.ModuleType("neat.trees.compartmenttree")
    compartmenttree_mod.CompartmentTree = object

    factorydefaults_mod = types.ModuleType("neat.factorydefaults")
    factorydefaults_mod.DefaultPhysiology = object

    class FakeH:
        def load_file(self, _name):
            return None

        def nrn_load_dll(self, _path):
            return None

    fake_neuron = types.ModuleType("neuron")
    fake_neuron.__version__ = "9.0.0"
    fake_neuron.h = FakeH()
    fake_neuron.load_mechanisms = lambda path: True

    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(module_name, module_path)
    )

    monkeypatch.setitem(sys.modules, "neat", neat_pkg)
    monkeypatch.setitem(sys.modules, "neat.simulations", simulations_pkg)
    monkeypatch.setitem(sys.modules, "neat.simulations.neuron", neuron_pkg)
    monkeypatch.setitem(sys.modules, "neat.trees", trees_pkg)
    monkeypatch.setitem(sys.modules, "neat.trees.morphtree", morphtree_mod)
    monkeypatch.setitem(sys.modules, "neat.trees.phystree", phystree_mod)
    monkeypatch.setitem(sys.modules, "neat.trees.compartmenttree", compartmenttree_mod)
    monkeypatch.setitem(sys.modules, "neat.factorydefaults", factorydefaults_mod)
    monkeypatch.setitem(sys.modules, "neuron", fake_neuron)
    monkeypatch.setitem(sys.modules, module_name, module)

    module.__spec__.loader.exec_module(module)

    return module


def test_load_neuron_model_raises_on_metadata_mismatch(monkeypatch):
    neuronmodel = _load_neuronmodel(monkeypatch)

    model_name = f"test_model_{uuid.uuid4().hex}"
    model_path = Path(neuronmodel.__file__).resolve().parent / "tmp" / model_name
    model_path.mkdir(parents=True)

    try:
        (model_path / "build_info.json").write_text(
            json.dumps(
                {
                    "neuron_version": "8.2.4",
                    "python_version": "3.13.0",
                    "python_executable": "/tmp/python",
                    "platform_system": "Darwin",
                    "platform_machine": "arm64",
                }
            )
        )

        with pytest.raises(
            neuronmodel.NeuronMechanismLoadError, match="built for a different runtime"
        ):
            neuronmodel.load_neuron_model(model_name)
    finally:
        shutil.rmtree(model_path)


def test_load_neuron_model_skips_duplicate_legacy_loads(monkeypatch):
    neuronmodel = _load_neuronmodel(monkeypatch)
    neuronmodel.neuron.__version__ = "8.2.4"

    calls = []

    def fake_nrn_load_dll(path):
        calls.append(path)

    monkeypatch.setattr(neuronmodel.h, "nrn_load_dll", fake_nrn_load_dll)

    model_name = f"test_model_{uuid.uuid4().hex}"
    model_path = Path(neuronmodel.__file__).resolve().parent / "tmp" / model_name
    lib_path = (
        model_path
        / neuronmodel.platform.machine()
        / ".libs"
        / "libnrnmech.so"
    )
    lib_path.parent.mkdir(parents=True)
    lib_path.touch()

    try:
        (model_path / "build_info.json").write_text(
            json.dumps(neuronmodel._get_neuron_runtime_metadata())
        )

        neuronmodel.load_neuron_model(model_name)
        neuronmodel.load_neuron_model(model_name)

        assert calls == [str(lib_path)]
    finally:
        shutil.rmtree(model_path)


def test_load_neuron_model_skips_duplicate_modern_loads(monkeypatch):
    neuronmodel = _load_neuronmodel(monkeypatch)

    calls = []

    def fake_load_mechanisms(path):
        calls.append(path)
        return True

    monkeypatch.setattr(neuronmodel.neuron, "load_mechanisms", fake_load_mechanisms)

    model_name = f"test_model_{uuid.uuid4().hex}"
    model_path = Path(neuronmodel.__file__).resolve().parent / "tmp" / model_name
    model_path.mkdir(parents=True)

    try:
        (model_path / "build_info.json").write_text(
            json.dumps(neuronmodel._get_neuron_runtime_metadata())
        )

        neuronmodel.load_neuron_model(model_name)
        neuronmodel.load_neuron_model(model_name)

        assert calls == [str(model_path)]
    finally:
        shutil.rmtree(model_path)
