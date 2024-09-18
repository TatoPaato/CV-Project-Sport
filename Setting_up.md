# Set Up the Project

The project was:

- Developed and tested on Ubuntu 24.04 LTS.
- Tested on Ubuntu 22.04.
- Tested on WSL2 - Ubuntu.

> **Note**:  
> In WSL2, to display interactive plots, ensure a backend such as `tk` is installed by running:
> 
> ```bash
> sudo apt install python3-tk
> ```

## Virtual Environments

The next steps are adapted from [this guide](https://linuxgenie.net/install-pip-ubuntu-24-04/).

> **Warning**:  
> Due to recent changes in Debian and Ubuntu 24.04, Python packages are not installed or managed via PIP in the global environment. If you use the pip command outside a Python environment, you may encounter the error: **externally managed environment**.  
> The recommended solution is to create a Python virtual environment for the project and manage the necessary packages there using **pip**.

### Installing the `venv` Module

Install the `venv` module using `apt`:

```bash
sudo apt install python3-venv
```

Got to the folder with all the project files and create a virtual environment ``<ve_name>`` for python and activate it using the source command:
```bash
python3 -m venv <ve_name>
source <ve_name>/bin/activate
```
You can now install the required packages by running

```bash 
pip install -r requirements.txt
```

***Note***: in Ubuntu 24.04 pip and pip3 commands refers both to python3-pip, since python2  is no longer officially supported.

**pip** can be installed globally by running.
```bash
sudo apt install python3-pip
```
