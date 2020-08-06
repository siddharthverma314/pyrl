{ buildPythonPackage
, pytorch
, gym
, cpprb
, flatten-dict
, termcolor
, pygments
, tabulate
}:
buildPythonPackage rec {
  pname = "pyrl";
  version = "0.1.0";

  src = builtins.path { name = pname; path = ./.; };

  propagatedBuildInputs = [
    pytorch
    gym
    cpprb
    flatten-dict
    termcolor
    pygments
    tabulate
  ];

  doCheck = false;
}
