# -*- coding: utf-8 -*-
#
# install.py
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

import os
import sys
import glob
import stat
import shutil
import inspect
import platform
from pathlib import Path
import importlib
import subprocess

try:
    import neuron
except ModuleNotFoundError:
    pass

from neat import IonChannel, ExpConcMech
from neat.simulations.nest import nestml_tools


def _allBaseClasses(cls):
    """
    Return list get all base classes from a given class
    """
    return [cls.__base__] + _allBaseClasses(cls.__base__) if cls is not None else []


class ChannelPathExtractor:
    def __init__(self, path_neat, model_name):
        self.path_neat = path_neat
        self.model_name = model_name

    def _extract_channel_path_and_modules(self, channel_path_arg):
        """
        Extract the path to the directory with the ".py" files containing ion
        ion channels, as well as a list of all ".py" modules that need to be scanned
        for ion channels.

        If the input path points to a single .py file, we will extract this .py file
        as a module. If the input path points to a directory, all .py files within
        will be loaded as modules and scanned for ion channels
        """
        # extract the channel path from arguments
        if self.model_name == "default":
            path_with_channels = os.path.join(
                self.path_neat, "channels/channelcollection/"
            )
        else:
            path_with_channels = channel_path_arg

        # parse the channel path
        if path_with_channels[-3:] == ".py":
            path_with_channels = path_with_channels.replace(".py", "")
            # path points to a single .py file, we load this file as a module
            path_with_channels, channel_module = os.path.split(path_with_channels)
            channel_modules = [channel_module]
        else:
            # path points to a directory, we search all files in the directory for
            # ion channels
            channel_modules = []
            for channel_module in glob.glob(os.path.join(path_with_channels, "*.py")):
                # import channel modules
                # convert names from glob to something susceptible to python import
                channel_module = os.path.split(channel_module)[1]
                channel_module = channel_module.replace(".py", "")
                channel_modules.append(channel_module)

        return path_with_channels, channel_modules

    def _collect_channels(self, path_with_channels, channel_modules):
        """
        Returns list with all channels found in the list of modules
        """
        sys.path.insert(0, path_with_channels)

        channels = []
        for channel_module in channel_modules:
            print(
                f"Reading channels from: "
                f"{os.path.join(path_with_channels, channel_module)}"
            )
            chans = importlib.import_module(channel_module)

            for name, obj in inspect.getmembers(chans):
                # if an object is a class and inheriting from IonChannel,
                # we append it to the channels list
                if inspect.isclass(obj) and IonChannel in _allBaseClasses(obj):
                    channels.append(obj())

        return channels

    def collect_channels(self, *channel_path_arg):
        """
        Collect all channels that can be found in the provided path arguments
        """
        channels = []
        for arg in channel_path_arg:
            channel_path, channel_modules = self._extract_channel_path_and_modules(arg)
            channels.extend(self._collect_channels(channel_path, channel_modules))

        return channels


def _check_model_name(model_name):
    if not len(model_name) > 0:
        raise IOError(
            "No model name [name] argument was provided. "
            "The model name can only be resolved automatically if exactly one "
            "[--path] argument is given."
        )

    if "/" in model_name or "." in model_name:
        raise IOError(
            "Model name [name] is a path name (contains '/') or "
            "a file name (contains '.', which is not allowed."
        )


def _resolve_model_name(model_name, channel_path_arg):
    if len(channel_path_arg) == 1:

        if model_name == "default":
            if len(channel_path_arg[0]) > 0:
                raise IOError(
                    "Model name [name] 'default' is reserved for the default "
                    "channel models, no path should be provided in "
                    "this case."
                )

        elif model_name == "":
            # the model name is not provided, but only a single path argument is
            # given. The model name is resolved as the last element in the
            # provided path
            path_aux = channel_path_arg[0].replace(".py", "")
            model_name = os.path.basename(os.path.normpath(path_aux))

        else:
            _check_model_name(model_name)

    else:
        _check_model_name(model_name)

    return model_name


def get_local_bin_dir():
    """
    Get the most local bin directory based on the active environment.
    Priority order:
    1. Conda environment bin
    2. Virtual environment bin (venv/virtualenv)
    3. Docker container /usr/local/bin (if in container)
    4. ~/.local/bin (fallback)

    Returns:
        Path: The bin directory to use
    """

    # 1. Check for conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        bin_dir = Path(conda_prefix) / "bin"
        if bin_dir.exists() and bin_dir.is_dir():
            env_name = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
            print(f"📦 Detected conda environment: {env_name}")
            return bin_dir

    # 2. Check for virtual environment (venv/virtualenv)
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        bin_dir = Path(virtual_env) / "bin"
        if bin_dir.exists() and bin_dir.is_dir():
            print(f"🐍 Detected Python virtual environment")
            return bin_dir

    # 3. Check if running in Docker container
    if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
        # In Docker, prefer /usr/local/bin for system-wide access
        bin_dir = Path("/usr/local/bin")
        if os.access(bin_dir, os.W_OK):
            print(f"🐳 Detected Docker container")
            print(f"   Using system bin (writable): {bin_dir}")
            return bin_dir
        else:
            # If /usr/local/bin is not writable, fall back to user location
            print(f"🐳 Detected Docker container")
            print(f"   /usr/local/bin not writable, using user bin")

    # 4. Fallback to ~/.local/bin
    print(f"ℹ️  No specific environment detected")
    return Path.home() / ".local" / "bin"


def create_nrnspecial_wrapper(
    special_path, wrapper_name="python_with_my_nrn_model", install_dir=None
):
    """
    Create a wrapper script for running Python with NEURON + CoreNEURON mechanisms

    Args:
        special_path: Path to the 'special' executable (e.g., /path/to/arm64/special)
        wrapper_name: Name for the wrapper script
        install_dir: Directory to install wrapper (default: auto-detect environment)
    """

    # Set default install directory based on environment
    if install_dir is None:
        install_dir = get_local_bin_dir()
        print(f"   Using directory: {install_dir}")
        print()
    else:
        install_dir = Path(install_dir)
        print(f"Using specified directory: {install_dir}")
        print()

    # Convert special_path to Path object and resolve to absolute path
    special_path = Path(special_path).resolve()

    # Check if special executable exists
    if not special_path.exists():
        print(f"Error: special executable not found at {special_path}", file=sys.stderr)
        sys.exit(1)

    if not special_path.is_file():
        print(f"Error: {special_path} is not a file", file=sys.stderr)
        sys.exit(1)

    # Create install directory if it doesn't exist
    if not install_dir.exists():
        print(f"Creating directory: {install_dir}")
        try:
            install_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created {install_dir}")
            print()
        except PermissionError:
            print(f"Error: Permission denied creating {install_dir}", file=sys.stderr)
            print(
                f"Try running with sudo or choose a different directory",
                file=sys.stderr,
            )
            sys.exit(1)

    # Check write permissions
    if not os.access(install_dir, os.W_OK):
        print(f"Error: No write permission for {install_dir}", file=sys.stderr)
        print(f"Try running with sudo or choose a different directory", file=sys.stderr)
        sys.exit(1)

    # Path for the wrapper script
    wrapper_path = install_dir / wrapper_name

    # Wrapper script content
    wrapper_content = f"""#!/bin/bash
# Wrapper to run Python scripts with NEURON + CoreNEURON mechanisms
# Auto-generated wrapper for: {special_path}

SPECIAL_PATH="{special_path}"

# Check if special exists
if [ ! -f "$SPECIAL_PATH" ]; then
    echo "Error: special executable not found at $SPECIAL_PATH" >&2
    exit 1
fi

# Run special with -python flag and pass all arguments
exec "$SPECIAL_PATH" -python "$@"
"""
    # Write the wrapper script
    try:
        with open(wrapper_path, "w") as f:
            f.write(wrapper_content)

        # Make it executable (chmod +x)
        wrapper_path.chmod(
            wrapper_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )

        print(f"✓ Created wrapper at: {wrapper_path}")
        print()

        # Check if install_dir is in PATH
        path_dirs = os.environ.get("PATH", "").split(":")
        if str(install_dir) in path_dirs:
            print(f"✓ {install_dir} is already in your PATH")
        else:
            print(f"⚠ Warning: {install_dir} is not in your current PATH")

            # Provide context-specific advice
            if os.environ.get("CONDA_PREFIX"):
                print(f"   This is unusual for an active conda environment.")
                print(
                    f"   Try: conda deactivate && conda activate {os.environ.get('CONDA_DEFAULT_ENV')}"
                )
            elif os.environ.get("VIRTUAL_ENV"):
                print(f"   This is unusual for an active virtual environment.")
                print(f"   Try deactivating and reactivating your environment.")
            elif os.path.exists("/.dockerenv"):
                print(f"   Add to your Dockerfile or docker run command:")
                print(f'   ENV PATH="{install_dir}:$PATH"')
            else:
                print(f"   Add to your ~/.bashrc or ~/.zshrc:")
                print(f'   export PATH="{install_dir}:$PATH"')

        return wrapper_path

    except Exception as e:
        print(f"Error creating wrapper: {e}", file=sys.stderr)
        sys.exit(1)


def _compile_neuron(model_name, path_neat, channels, path_neuronresource=None):

    print(f"path of this file: {__file__}   ")

    # combine `model_name` with the neuron compilation path
    path_for_neuron_compilation = os.path.join(
        path_neat, "simulations/neuron/tmp/", model_name
    )
    # delete old compiled files if exist
    if os.path.exists(path_for_neuron_compilation):
        shutil.rmtree(path_for_neuron_compilation)
    path_for_mod_files = os.path.join(path_for_neuron_compilation, "mech/")

    print(f"--- writing channels to \n" f" > {path_for_mod_files}")

    # Create the "mech/" directory in a clean state
    os.makedirs(path_for_mod_files)

    # copy mechanisms from resource path
    if path_neuronresource is not None:
        for mod_file in glob.glob(os.path.join(path_neuronresource, "*.mod")):
            shutil.copy2(mod_file, path_for_mod_files)

    for chan in channels:
        print(" - writing .mod file for:", chan.__class__.__name__)
        chan.write_mod_file(path_for_mod_files)

    # change to directory where 'mech/' folder is located and compile the mechanisms
    os.chdir(path_for_neuron_compilation)
    # with open(".noindex", "w") as f:  # prevent mac from indexing compiled files
    #     pass
    # subprocess.run(["mdutil", "-i", "off", path_for_neuron_compilation])
    # if os.path.exists(f"{platform.machine()}/"):  # delete old compiled files if exist
    #     shutil.rmtree(f"{platform.machine()}/")
    print("!!!", os.getcwd())
    # my_env = os.environ.copy()
    # # This forces every sub-call to 'make' to append -j1, effectively
    # # overriding the -j4 passed by the nrnivmodl wrapper.
    # my_env["MAKE"] = "make -j1"

    # # Also keep these for good measure
    # my_env["MAKEFLAGS"] = "-j1"
    if int(neuron.__version__.split(".")[0]) < 9:
        subprocess.call(["nrnivmodl", "mech/"])  # compile all mod files
    else:
        subprocess.call(["nrnivmodl", "-coreneuron", "mech/"])  # compile all mod files

    create_nrnspecial_wrapper(
        special_path=os.path.join(
            path_for_neuron_compilation,
            f"{platform.machine()}/special",
        ),
        wrapper_name=f"python_with_{model_name}",
    )

    print(
        f"\n------------------------------\n"
        f"The compiled .mod-files can be loaded into neuron using:\n"
        f'    neat.load_neuron_model("{model_name}")\n\n'
        f"If you want to use the compiled .mod-files with CoreNEURON, use:\n"
        f"    python_with_{model_name} my_script.py \n"
        f"------------------------------\n"
    )


def _compile_nest(model_name, path_neat, channels, path_nestresource=None, ions=["ca"]):
    from pynestml.frontend.pynestml_frontend import generate_nest_compartmental_target

    # assert that `model_name` is a pure name
    assert not "/" in model_name
    assert not "." in model_name

    # combine `model_name` with the nestml compilation path
    path_for_nestml_compilation = os.path.join(
        path_neat, "simulations/nest/tmp/", model_name
    )

    # Create the model directory in a clean state
    if os.path.exists(path_for_nestml_compilation):
        shutil.rmtree(path_for_nestml_compilation)
    os.makedirs(path_for_nestml_compilation)

    print(f"--- writing nestml model to \n" f"    > {path_for_nestml_compilation}")

    if path_nestresource is not None:
        blocks = nestml_tools.parse_nestml_file(path_nestresource)

    for chan in channels:
        print(" - writing .nestml blocks for:", chan.__class__.__name__)
        blocks_ = chan.write_nestml_blocks(v_comp=-75.0)

        for block, blockstr in blocks_.items():
            blocks[block] = blockstr + blocks[block]

    for ion in ions:
        concmech = ExpConcMech(ion)
        blocks_ = concmech.write_nestml_blocks(channels=channels)

        for block, blockstr in blocks_.items():
            blocks[block] = blocks[block] + blockstr

    # create directory to install nestml files
    if not os.path.exists(path_for_nestml_compilation):
        os.makedirs(path_for_nestml_compilation)
    # write the nestml file
    nestml_file_path = nestml_tools.write_nestml_blocks(
        blocks, path_for_nestml_compilation, model_name + "_model", v_comp=-75.0
    )

    generate_nest_compartmental_target(
        input_path=nestml_file_path,
        target_path=path_for_nestml_compilation,
        module_name=model_name + "_module",
        logging_level="DEBUG",
    )


def _install_models(
    model_name,
    path_neat,
    channel_path_arg,
    simulators=["neuron", "nest"],
    path_nestresource=None,
    path_neuronresource=None,
):
    """
    Compile a set of ion channels models specified by [channel_path_arg]

    Parameters
    ----------
    model_name: str
        The name of the compiled model that can be used to load it with
        `neat.load_neuron_model()` or `neat.load_nest_model()`
    path_neat: str
        The path to the root directory of the imported neat module
    channel_path_arg: list of str
        Path argument to the channel files, to be parsed by `ChannelPathExtractor`
    simulators: list of str
        The simulators for which to compile the channels
    path_nestresource: str
        Optional NESTML file containing for instance synaptic receptors, will
        be combined with the channels into a single .nestml file
    path_neuronresource: str
        Optional path to a directory with .mod files, these modfiles will be
        copied to the NEURON install directory and compiled together with the
        generated channel .mod files
    """
    model_name = _resolve_model_name(model_name, channel_path_arg)

    # collect the ion channels from the provide path arguments
    cpex = ChannelPathExtractor(path_neat, model_name)
    channels = cpex.collect_channels(*channel_path_arg)

    if "neuron" in simulators:
        _compile_neuron(
            model_name, path_neat, channels, path_neuronresource=path_neuronresource
        )
    if "nest" in simulators:
        _compile_nest(
            model_name, path_neat, channels, path_nestresource=path_nestresource
        )
