# Personal repository for Deep Learning Experiments

Conveniently named "dl", this repository is created for having a
living infrastructure for studying **deep learning** literature as
well as trying out ideas. As it's a personal, small-scale effort
it was not designed with large-scale deep learning best practices
in mind such as distributed training. It aims to keep a
neat code base with minimal functionality, while documenting
rationale and subtleties of the algorithms as much as possible for
future reference.

# Structure

Following are the top level modules and their explanations.

```
core/
Contains code for essential deep learning setup such as train/eval
tasks, experiments, "assembled" model (that consists of backbone
and heads) etc.

data_types/
Collects commonly used data types among deep learning tasks, aiming
strong typing instead of indexed iterable access.

interfaces/
Common interfaces each project should implement to benefit from the
core module.

projects/
Projects, usually characterized by a dataset and an architecture.

tools/
Machinery for interaction with the codebase, e.g. container management,
starting experiments, running tests and checks. Divided into host,
container and common, depending on which environment the code is run.

zoo/
Implementation of deep learning architectures, both at block level
and model level.
```

Module tests reside under `_test` directory as they are private to the
module and the leading underscore makes it easier to distinguish the
tests from the module code.
