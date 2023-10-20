<a href="https://github.com/alexandrainst/doms_databasen"><img src="gfx/alexandra_logo.png" width="239" height="175" align="right" /></a>
# doms_databasen

En scrapet udgave af [domsdatabasen](https://domsdatabasen.dk/#), der for nyligt er udgivet som åben data.


Hver enkelt dom kan tilgås via https://domsdatabasen.dk/#sag/\<nummer\>, hvor \<nummer\> er mellem 1 og 3821 (pr. 11-10-2023).


______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/doms_databasen/doms_databasen.html)
[![License](https://img.shields.io/github/license/alexandrainst/doms_databasen)](https://github.com/alexandrainst/doms_databasen/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/doms_databasen)](https://github.com/alexandrainst/doms_databasen/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)](https://github.com/alexandrainst/doms_databasen/tree/main/tests)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/doms_databasen/blob/main/CODE_OF_CONDUCT.md)


Developers:

- Oliver Kinch (oliver.kinch@alexandra.dk)
- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)


## Setup

### Installation

1. Run `pip install -r requirements.txt` to install the required packages.


## A Word on Modules and Scripts
In the `src` directory there are two subdirectories, `doms_databasen`
and `scripts`. This is a brief explanation of the differences between the two.

### Modules
All Python files in the `doms_databasen` directory are _modules_
internal to the project package. Examples here could be a general data loading script,
a definition of a model, or a training function. Think of modules as all the building
blocks of a project.

When a module is importing functions/classes from other modules we use the _relative
import_ notation - here's an example:

```
from .other_module import some_function
```

### Scripts
Python files in the `scripts` folder are scripts, which are short code snippets that
are _external_ to the project package, and which is meant to actually run the code. As
such, _only_ scripts will be called from the terminal. An analogy here is that the
internal `numpy` code are all modules, but the Python code you write where you import
some `numpy` functions and actually run them, that a script.

When importing module functions/classes when you're in a script, you do it like you
would normally import from any other package:

```
from doms_databasen import some_function
```

Note that this is also how we import functions/classes in tests, since each test Python
file is also a Python script, rather than a module.


## Features

### Automatic Test Coverage Calculation

Run `make test` to test your code, which also updates the "coverage badge" in the
README, showing you how much of your code base that is currently being tested.

### Continuous Integration

Github CI pipelines are included in the repo, running all the tests in the `tests`
directory, as well as building online documentation, if Github Pages has been enabled
for the repository (can be enabled on Github in the repository settings).

### Code Spaces

Code Spaces is a new feature on Github, that allows you to develop on a project
completely in the cloud, without having to do any local setup at all. This repo comes
included with a configuration file for running code spaces on Github. When hosted on
`alexandrainst/doms_databasen` then simply press the `<> Code` button
and add a code space to get started, which will open a VSCode window directly in your
browser.


## Project structure
```
.
├── .devcontainer
│   └── devcontainer.json
├── .github
│   └── workflows
│       ├── ci.yaml
│       └── docs.yaml
├── .gitignore
├── .name_and_email
├── .pre-commit-config.yaml
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── README.md
├── config
│   ├── __init__.py
│   ├── config.yaml
│   └── hydra
│       └── job_logging
│           └── custom.yaml
├── data
├── gfx
│   └── alexandra_logo.png
├── makefile
├── models
├── notebooks
├── poetry.toml
├── pyproject.toml
├── src
│   ├── scripts
│   │   ├── fix_dot_env_file.py
│   │   └── your_script.py
│   └── doms_databasen
│       ├── __init__.py
│       └── your_module.py
└── tests
    ├── __init__.py
    └── test_dummy.py
```
