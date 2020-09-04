{ buildPythonPackage
, pytorch
, gym
, cpprb
, flatten-dict
, termcolor
, pygments
, tabulate
, nix-gitignore
}:
buildPythonPackage rec {
  pname = "pyrl";
  version = "0.1.0";

  src = nix-gitignore.gitignoreSource [] ./.;

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
